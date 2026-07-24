package grove

import (
	"errors"
	"math"
	"math/rand"
)

// Fit trains a gradient-boosted ensemble on X (row-major, n×d) and labels y.
// For Binary, y holds 0/1; for Multiclass, y holds class indices in [0,NumClass).
// Unset Params fields take the defaults from Default. The fit is deterministic
// for a given Seed.
func Fit(X [][]float64, y []float64, p Params) (*Model, error) {
	if err := p.fill(); err != nil {
		return nil, err
	}
	n := len(X)
	if n == 0 || len(y) != n {
		return nil, errors.New("grove: empty dataset or len(X) != len(y)")
	}
	d := len(X[0])

	k := 1
	if p.Objective == Multiclass {
		k = p.NumClass
		for _, yi := range y {
			if yi < 0 || int(yi) >= k {
				return nil, errors.New("grove: label out of range for NumClass")
			}
		}
	}

	bnr := fitBinner(X, p.MaxBins)
	bt := bnr.transpose(X)
	base := baseScores(y, k, n)

	// raw[i] holds the running per-class score for sample i (init to base).
	raw := make([][]float64, n)
	for i := range raw {
		raw[i] = append([]float64(nil), base...)
	}

	m := &Model{
		Objective:    p.Objective,
		NumClass:     k,
		NumFeature:   d,
		Base:         base,
		LearningRate: p.LearningRate,
	}
	tp := treeParams{
		maxDepth:       p.MaxDepth,
		lambda:         p.Lambda,
		gamma:          p.Gamma,
		minChildWeight: p.MinChildWeight,
		lr:             p.LearningRate,
	}

	g := make([]float64, n)
	h := make([]float64, n)
	prob := make([]float64, k)
	all := iota0(n)
	rng := rand.New(rand.NewSource(p.Seed))

	for round := 0; round < p.Rounds; round++ {
		trees := make([]tree, k)
		for c := 0; c < k; c++ {
			gradients(raw, y, c, k, prob, g, h)
			idx := all
			if p.Subsample < 1 {
				idx = subsample(all, p.Subsample, rng)
			}
			t := buildTree(bt, bnr.edges, g, h, idx, tp)
			for i := 0; i < n; i++ {
				raw[i][c] += t.predict(X[i])
			}
			trees[c] = t
		}
		m.Rounds = append(m.Rounds, trees)
	}
	return m, nil
}

// baseScores returns the initial per-class raw score (the constant model): the
// label log-odds for Binary, per-class log-frequency for Multiclass.
func baseScores(y []float64, k, n int) []float64 {
	if k == 1 {
		var pos float64
		for _, yi := range y {
			pos += yi
		}
		mean := clamp(pos/float64(n), 1e-6, 1-1e-6)
		return []float64{math.Log(mean / (1 - mean))}
	}
	cnt := make([]float64, k)
	for _, yi := range y {
		cnt[int(yi)]++
	}
	base := make([]float64, k)
	for c := range base {
		base[c] = math.Log(clamp(cnt[c]/float64(n), 1e-6, 1))
	}
	return base
}

// gradients fills g,h with the first/second derivatives of the loss w.r.t. the
// raw score of class c (logistic for Binary, softmax for Multiclass).
func gradients(raw [][]float64, y []float64, c, k int, prob, g, h []float64) {
	const eps = 1e-6
	if k == 1 {
		for i := range raw {
			p := sigmoid(raw[i][0])
			g[i] = p - y[i]
			h[i] = math.Max(p*(1-p), eps)
		}
		return
	}
	for i := range raw {
		softmax(raw[i], prob)
		yc := 0.0
		if int(y[i]) == c {
			yc = 1
		}
		g[i] = prob[c] - yc
		h[i] = math.Max(prob[c]*(1-prob[c]), eps)
	}
}
