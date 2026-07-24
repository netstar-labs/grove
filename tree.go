package grove

import (
	"runtime"
	"sync"
)

// node is one vertex of a regression tree. Internal nodes route on
// "feature <= threshold" (left) or otherwise (right); leaves carry a weight.
// The compact JSON tags keep serialized ensembles small.
type node struct {
	Leaf        bool    `json:"leaf,omitempty"`
	Feature     int     `json:"f,omitempty"`
	Threshold   float64 `json:"t,omitempty"`
	Left        int     `json:"l,omitempty"`
	Right       int     `json:"r,omitempty"`
	Value       float64 `json:"v,omitempty"`
	Gain        float64 `json:"g,omitempty"`  // split gain, for feature importance
	DefaultLeft bool    `json:"dl,omitempty"` // route missing (NaN) features left
}

// tree is a flat, index-linked node array. Node 0 is the root.
type tree struct {
	Nodes []node `json:"nodes"`
}

// predict routes a raw feature vector to its leaf and returns the leaf weight.
// A missing feature (NaN) follows the node's learned default direction.
func (t *tree) predict(x []float64) float64 {
	i := 0
	for !t.Nodes[i].Leaf {
		n := &t.Nodes[i]
		v := x[n.Feature]
		goLeft := n.DefaultLeft
		if v == v { // not NaN
			goLeft = v <= n.Threshold
		}
		if goLeft {
			i = n.Left
		} else {
			i = n.Right
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

// Building a node's histogram fans out across workers only when it pays: at
// least parThreshold samples (below it, goroutine overhead exceeds the work) and
// at least minParFeatures features (few features ⇒ per-worker work too small, so
// narrow fits stay fully serial and never pay the concurrency tax).
const (
	parThreshold   = 2048
	minParFeatures = 16
)

// arena pools flat per-node histogram buffers (each d*maxNB, one for gradients
// one for hessians) so histogram subtraction reuses memory across the whole
// forest instead of allocating per node. One arena is shared across all trees of
// a fit; the recursion is single-threaded, so the free list needs no lock.
type arena struct {
	d, maxNB, size int
	free           [][2][]float64
}

func newArena(edges [][]float64) *arena {
	maxNB := 1
	for _, e := range edges {
		if nb := len(e) + 2; nb > maxNB { // +1 regular top bin, +1 missing bin
			maxNB = nb
		}
	}
	d := len(edges)
	return &arena{d: d, maxNB: maxNB, size: d * maxNB}
}

func (a *arena) get() (hg, hh []float64) {
	if n := len(a.free); n > 0 {
		buf := a.free[n-1]
		a.free = a.free[:n-1]
		return buf[0], buf[1]
	}
	return make([]float64, a.size), make([]float64, a.size)
}

func (a *arena) put(hg, hh []float64) { a.free = append(a.free, [2][]float64{hg, hh}) }

// builder grows one tree. A node's histogram (feature-major, flat with stride
// maxNB) is built once; on a split it builds only the smaller child's histogram
// and derives the larger by subtracting from the parent's — roughly halving the
// histogram-building work.
type builder struct {
	bt       [][]uint8
	edges    [][]float64
	g, h     []float64
	p        treeParams
	t        *tree
	ar       *arena
	nWorkers int
}

// buildTree grows a single regression tree over idx, taking histogram buffers
// from the shared arena.
func buildTree(bt [][]uint8, edges [][]float64, g, h []float64, idx []int, p treeParams, ar *arena) tree {
	nWorkers := 1
	if ar.d >= minParFeatures {
		if nWorkers = min(runtime.GOMAXPROCS(0), ar.d); nWorkers < 1 {
			nWorkers = 1
		}
	}
	b := &builder{bt: bt, edges: edges, g: g, h: h, p: p, t: &tree{}, ar: ar, nWorkers: nWorkers}
	hg, hh := ar.get()
	b.buildHist(idx, hg, hh)
	b.grow(idx, 0, hg, hh)
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

// grow adds the subtree for idx (whose histogram is hg/hh) at the given depth,
// returns its root node index, and returns hg/hh to the arena when done with them.
func (b *builder) grow(idx []int, depth int, hg, hh []float64) int {
	// node totals: feature 0 occupies indices [0,maxNB); every sample lands in
	// one of its bins, so summing that region is the node's (G,H).
	var G, H float64
	for i := 0; i < b.ar.maxNB; i++ {
		G += hg[i]
		H += hh[i]
	}
	self := len(b.t.Nodes)
	b.t.Nodes = append(b.t.Nodes, node{}) // reserve; filled below

	leaf := func() int {
		b.t.Nodes[self] = node{Leaf: true, Value: b.leafValue(G, H)}
		b.ar.put(hg, hh)
		return self
	}
	if depth >= b.p.maxDepth || len(idx) < 2 {
		return leaf()
	}

	bestF, bestBin, bestGain, defaultLeft := b.bestSplit(hg, hh, G, H)
	if bestF < 0 {
		return leaf()
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

	// build only the smaller child's histogram; derive the larger by subtracting
	// it from the parent's (shg/shh hold the smaller, lhg/lhh the larger).
	shg, shh := b.ar.get()
	lhg, lhh := b.ar.get()
	small := left
	if len(right) < len(left) {
		small = right
	}
	b.buildHist(small, shg, shh)
	subInto(lhg, hg, shg)
	subInto(lhh, hh, shh)
	b.ar.put(hg, hh) // parent histogram no longer needed

	// route the smaller/larger histograms to the correct left/right child.
	var lIdx, rIdx int
	if len(right) < len(left) { // smaller == right
		rIdx = b.grow(right, depth+1, shg, shh)
		lIdx = b.grow(left, depth+1, lhg, lhh)
	} else { // smaller == left
		lIdx = b.grow(left, depth+1, shg, shh)
		rIdx = b.grow(right, depth+1, lhg, lhh)
	}
	b.t.Nodes[self] = node{
		Feature:     bestF,
		Threshold:   b.edges[bestF][bestBin],
		Left:        lIdx,
		Right:       rIdx,
		Gain:        bestGain,
		DefaultLeft: defaultLeft,
	}
	return self
}

// buildHist accumulates the (gradient, hessian) histogram of idx into hg/hh
// (feature-major, stride maxNB), clearing them first. Large nodes on wide
// feature sets fan out across workers, each owning a disjoint stripe of features
// — and therefore disjoint index ranges of hg/hh — so the fill is race-free.
func (b *builder) buildHist(idx []int, hg, hh []float64) {
	clear(hg)
	clear(hh)
	if b.nWorkers <= 1 || len(idx) < parThreshold {
		for f := 0; f < b.ar.d; f++ {
			b.fillFeature(f, idx, hg, hh)
		}
		return
	}
	var wg sync.WaitGroup
	chunk := (b.ar.d + b.nWorkers - 1) / b.nWorkers
	for w := 0; w < b.nWorkers; w++ {
		lo, hi := w*chunk, min((w+1)*chunk, b.ar.d)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for f := lo; f < hi; f++ {
				b.fillFeature(f, idx, hg, hh)
			}
		}(lo, hi)
	}
	wg.Wait()
}

func (b *builder) fillFeature(f int, idx []int, hg, hh []float64) {
	off := f * b.ar.maxNB
	col := b.bt[f]
	for _, i := range idx {
		bin := off + int(col[i])
		hg[bin] += b.g[i]
		hh[bin] += b.h[i]
	}
}

// subInto sets dst = a - b element-wise (the larger child's histogram).
func subInto(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// bestSplit scans the node's pre-built histograms for the (feature, bin)
// maximizing the regularized gain, and the direction missing values should take.
// Cheap (O(d·maxNB)) since the histograms are already built, so it stays serial;
// the cost was the fill, now parallel. G/H are the node totals (they include the
// missing bin), so for each split point it tries routing the missing mass to
// each side and keeps the better. With no missing samples both are equal, so a
// clean dataset yields the same split as if the feature had no missing bin.
func (b *builder) bestSplit(hg, hh []float64, G, H float64) (bestF, bestBin int, bestGain float64, defaultLeft bool) {
	bestF, bestBin, bestGain, defaultLeft = -1, 0, b.p.gamma, true
	for f := 0; f < b.ar.d; f++ {
		nbReg := len(b.edges[f]) + 1 // regular bins 0..nbReg-1
		if nbReg < 2 {
			continue
		}
		off := f * b.ar.maxNB
		Gm, Hm := hg[off+nbReg], hh[off+nbReg] // the missing bin
		var GL, HL float64
		for bin := 0; bin < nbReg-1; bin++ {
			GL += hg[off+bin]
			HL += hh[off+bin]
			// missing left: regular-left plus the missing mass
			if g, ok := b.gainOf(GL+Gm, HL+Hm, G, H); ok && g > bestGain {
				bestF, bestBin, bestGain, defaultLeft = f, bin, g, true
			}
			// missing right: regular-left only (missing falls to the right)
			if g, ok := b.gainOf(GL, HL, G, H); ok && g > bestGain {
				bestF, bestBin, bestGain, defaultLeft = f, bin, g, false
			}
		}
	}
	return
}

// gainOf is the regularized gain of a split with the given left sums (right is
// the node total minus left), or ok=false if either child is under-weight.
func (b *builder) gainOf(GL, HL, G, H float64) (float64, bool) {
	GR, HR := G-GL, H-HL
	if HL < b.p.minChildWeight || HR < b.p.minChildWeight {
		return 0, false
	}
	return 0.5 * (score(GL, HL, b.p.lambda) + score(GR, HR, b.p.lambda) - score(G, H, b.p.lambda)), true
}
