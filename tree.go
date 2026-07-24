package grove

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

// builder grows one tree over a sample-index set using precomputed bins and the
// current gradients/hessians.
type builder struct {
	bt    [][]uint8   // feature-major bins
	edges [][]float64 // per-feature thresholds (bin -> real value)
	g, h  []float64   // per-sample gradient and hessian
	p     treeParams
	t     *tree
}

// buildTree grows a single regression tree over the given sample indices.
func buildTree(bt [][]uint8, edges [][]float64, g, h []float64, idx []int, p treeParams) tree {
	b := &builder{bt: bt, edges: edges, g: g, h: h, p: p, t: &tree{}}
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

	bestF, bestBin, bestGain := -1, 0, b.p.gamma
	for f := range b.bt {
		nb := b.nBinsForNode(f)
		if nb < 2 {
			continue
		}
		hg := make([]float64, nb)
		hh := make([]float64, nb)
		for _, i := range idx {
			bin := b.bt[f][i]
			hg[bin] += b.g[i]
			hh[bin] += b.h[i]
		}
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
				bestF, bestBin, bestGain = f, bin, gain
			}
		}
	}

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

func (b *builder) nBinsForNode(f int) int { return len(b.edges[f]) + 1 }
