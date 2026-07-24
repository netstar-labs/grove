package grove

import (
	"math"
	"runtime"
	"sync"
)

// node is one vertex of a regression tree. Internal nodes route on
// "feature <= threshold" (left) or otherwise (right); leaves carry a weight.
// The compact JSON tags keep serialized ensembles small.
type node struct {
	Leaf      bool    `json:"leaf,omitempty"`
	Feature   int     `json:"f,omitempty"`
	Threshold float64 `json:"t,omitempty"`
	Left      int     `json:"l,omitempty"`
	Right     int     `json:"r,omitempty"`
	Value     float64 `json:"v,omitempty"`
	Gain      float64 `json:"g,omitempty"` // split gain, for feature importance
}

// tree is a flat, index-linked node array. Node 0 is the root.
type tree struct {
	Nodes []node `json:"nodes"`
}

// predict routes a raw feature vector to its leaf and returns the leaf weight.
func (t *tree) predict(x []float64) float64 {
	i := 0
	for !t.Nodes[i].Leaf {
		if x[t.Nodes[i].Feature] <= t.Nodes[i].Threshold {
			i = t.Nodes[i].Left
		} else {
			i = t.Nodes[i].Right
		}
	}
	return t.Nodes[i].Value
}

// treeParams are the per-tree hyperparameters the builder needs.
type treeParams struct {
	maxDepth       int
	lambda         float64
	gamma          float64
	minChildWeight float64
	lr             float64
}

// Split finding fans out across workers only when it pays: a node needs at
// least parThreshold samples (below it, goroutine overhead exceeds the
// histogram work), and the model needs at least minParFeatures features (with
// few features, per-worker work is too small — narrow fits stay fully serial,
// so they never pay the concurrency tax).
const (
	parThreshold   = 2048
	minParFeatures = 16
)

// splitCand is one feature's best split, filled by a worker and reduced in
// feature order (so the parallel result is identical to the serial one).
type splitCand struct {
	bin  int
	gain float64
}

// builder grows one tree over a sample-index set using precomputed bins and the
// current gradients/hessians. Its scratch (per-worker histograms, per-feature
// results) is allocated once and reused for every node.
type builder struct {
	bt       [][]uint8   // feature-major bins
	edges    [][]float64 // per-feature thresholds (bin -> real value)
	g, h     []float64   // per-sample gradient and hessian
	p        treeParams
	t        *tree
	nWorkers int
	hgW, hhW [][]float64 // per-worker histogram scratch [nWorkers][maxBins]
	res      []splitCand // per-feature best split (reduced in order)
}

// buildTree grows a single regression tree over the given sample indices,
// fanning split finding across up to GOMAXPROCS workers on large nodes.
func buildTree(bt [][]uint8, edges [][]float64, g, h []float64, idx []int, p treeParams) tree {
	maxNB := 1
	for _, e := range edges {
		if nb := len(e) + 1; nb > maxNB {
			maxNB = nb
		}
	}
	nWorkers := 1
	if len(edges) >= minParFeatures {
		if nWorkers = min(runtime.GOMAXPROCS(0), len(edges)); nWorkers < 1 {
			nWorkers = 1
		}
	}
	b := &builder{
		bt: bt, edges: edges, g: g, h: h, p: p, t: &tree{},
		nWorkers: nWorkers,
		hgW:      make([][]float64, nWorkers),
		hhW:      make([][]float64, nWorkers),
	}
	for w := range b.hgW {
		b.hgW[w] = make([]float64, maxNB)
		b.hhW[w] = make([]float64, maxNB)
	}
	if nWorkers > 1 {
		b.res = make([]splitCand, len(edges)) // per-feature results, parallel path only
	}
	b.grow(idx, 0)
	return *b.t
}

// score is the regularized similarity term G²/(H+λ) that appears three times in
// the split gain (parent and both children). Distinct from a leaf's weight.
func score(G, H, lambda float64) float64 { return G * G / (H + lambda) }

// leafValue is the regularized optimal weight for a leaf: -G/(H+λ), shrunk by
// the learning rate.
func (b *builder) leafValue(G, H float64) float64 {
	return -G / (H + b.p.lambda) * b.p.lr
}

// grow adds the subtree for sample set idx at the given depth and returns the
// index of its root node.
func (b *builder) grow(idx []int, depth int) int {
	var G, H float64
	for _, i := range idx {
		G += b.g[i]
		H += b.h[i]
	}
	self := len(b.t.Nodes)
	b.t.Nodes = append(b.t.Nodes, node{}) // reserve; filled below

	if depth >= b.p.maxDepth || len(idx) < 2 {
		b.t.Nodes[self] = node{Leaf: true, Value: b.leafValue(G, H)}
		return self
	}

	bestF, bestBin, bestGain := b.bestSplit(idx, G, H)
	if bestF < 0 {
		b.t.Nodes[self] = node{Leaf: true, Value: b.leafValue(G, H)}
		return self
	}

	left := make([]int, 0, len(idx))
	right := make([]int, 0, len(idx))
	for _, i := range idx {
		if int(b.bt[bestF][i]) <= bestBin {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	l := b.grow(left, depth+1)
	r := b.grow(right, depth+1)
	b.t.Nodes[self] = node{
		Feature:   bestF,
		Threshold: b.edges[bestF][bestBin],
		Left:      l,
		Right:     r,
		Gain:      bestGain,
	}
	return self
}

// bestSplit finds the (feature, bin) maximizing gain over idx. Large nodes fan
// out across workers, each scanning a stripe of features into its own scratch;
// the per-feature bests are then reduced in feature order, so the parallel and
// serial results are identical (and the reduction is deterministic).
func (b *builder) bestSplit(idx []int, G, H float64) (bestF, bestBin int, bestGain float64) {
	bestF, bestBin, bestGain = -1, 0, b.p.gamma
	d := len(b.bt)

	if b.nWorkers <= 1 || len(idx) < parThreshold {
		for f := 0; f < d; f++ {
			bin, gain := b.featureBest(f, idx, G, H, b.hgW[0], b.hhW[0])
			if gain > bestGain {
				bestF, bestBin, bestGain = f, bin, gain
			}
		}
		return
	}

	var wg sync.WaitGroup
	chunk := (d + b.nWorkers - 1) / b.nWorkers
	for w := 0; w < b.nWorkers; w++ {
		lo, hi := w*chunk, min((w+1)*chunk, d)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			for f := lo; f < hi; f++ {
				bin, gain := b.featureBest(f, idx, G, H, b.hgW[w], b.hhW[w])
				b.res[f] = splitCand{bin, gain}
			}
		}(w, lo, hi)
	}
	wg.Wait()

	for f := 0; f < d; f++ {
		if b.res[f].gain > bestGain {
			bestF, bestBin, bestGain = f, b.res[f].bin, b.res[f].gain
		}
	}
	return
}

// featureBest builds feature f's gradient/hessian histogram over idx (into the
// caller's scratch) and returns its best split bin and gain (gain −Inf if the
// feature can't split).
func (b *builder) featureBest(f int, idx []int, G, H float64, hg, hh []float64) (int, float64) {
	nb := len(b.edges[f]) + 1
	if nb < 2 {
		return 0, math.Inf(-1)
	}
	hg, hh = hg[:nb], hh[:nb]
	clear(hg)
	clear(hh)
	for _, i := range idx {
		bin := b.bt[f][i]
		hg[bin] += b.g[i]
		hh[bin] += b.h[i]
	}
	bestBin, bestGain := 0, math.Inf(-1)
	var GL, HL float64
	for bin := 0; bin < nb-1; bin++ {
		GL += hg[bin]
		HL += hh[bin]
		GR, HR := G-GL, H-HL
		if HL < b.p.minChildWeight || HR < b.p.minChildWeight {
			continue
		}
		gain := 0.5 * (score(GL, HL, b.p.lambda) + score(GR, HR, b.p.lambda) - score(G, H, b.p.lambda))
		if gain > bestGain {
			bestBin, bestGain = bin, gain
		}
	}
	return bestBin, bestGain
}
