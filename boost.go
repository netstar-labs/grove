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
	} else {
		for _, yi := range y {
			if yi != 0 && yi != 1 {
				return nil, errors.New("grove: binary labels must be 0 or 1")
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
		Version:      modelVersion,
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

	// optional train/validation split for early stopping (val rows never enter a
	// tree; their running score is kept only to measure held-out loss).
	trainIdx := iota0(n)
	var valIdx []int
	if p.EarlyStop > 0 && p.ValFraction > 0 && n >= 4 {
		sp := iota0(n)
		sr := rand.New(rand.NewSource(p.Seed ^ 0x9e3779b9))
		sr.Shuffle(n, func(i, j int) { sp[i], sp[j] = sp[j], sp[i] })
		nv := clampInt(int(p.ValFraction*float64(n)), 1, n-1)
		valIdx, trainIdx = sp[:nv], sp[nv:]
	}

	prob := make([]float64, k)
	perm := append([]int(nil), trainIdx...)
	nt := len(trainIdx)
	ks := nt
	if p.Subsample < 1 {
		if ks = int(p.Subsample * float64(nt)); ks < 1 {
			ks = 1
		}
	}
	rng := rand.New(rand.NewSource(p.Seed))
	sample := func() []int {
		if p.Subsample >= 1 {
			return perm
		}
		return partialSample(perm, ks, rng)
	}

	// gradient/hessian scratch: one pair for binary; per-class arrays for
	// multiclass, so a round computes softmax once per sample (O(n·k)) rather
	// than once per (sample, class) (O(n·k²)).
	g := make([]float64, n)
	h := make([]float64, n)
	var gc, hc [][]float64
	if k > 1 {
		gc = make([][]float64, k)
		hc = make([][]float64, k)
		for c := range gc {
			gc[c] = make([]float64, n)
			hc[c] = make([]float64, n)
		}
	}

	bestRound, bestLoss, since := -1, math.Inf(1), 0
	for round := 0; round < p.Rounds; round++ {
		trees := make([]tree, k)
		if k == 1 {
			for i := range raw {
				g[i], h[i] = gradHess(sigmoid(raw[i][0]), y[i])
			}
			t := buildTree(bt, bnr.edges, g, h, sample(), tp)
			for i := range raw {
				raw[i][0] += t.predict(X[i])
			}
			trees[0] = t
		} else {
			for i := range raw {
				softmax(raw[i], prob)
				yi := int(y[i])
				for c := 0; c < k; c++ {
					yc := 0.0
					if yi == c {
						yc = 1
					}
					gc[c][i], hc[c][i] = gradHess(prob[c], yc)
				}
			}
			for c := 0; c < k; c++ {
				t := buildTree(bt, bnr.edges, gc[c], hc[c], sample(), tp)
				for i := range raw {
					raw[i][c] += t.predict(X[i])
				}
				trees[c] = t
			}
		}
		m.Rounds = append(m.Rounds, trees)

		if p.EarlyStop > 0 {
			loss := valLoss(raw, y, valIdx, k)
			if loss < bestLoss-1e-12 {
				bestLoss, bestRound, since = loss, round, 0
			} else if since++; since >= p.EarlyStop {
				break
			}
		}
	}
	// early stopping keeps only the trees up to the best validation round.
	if p.EarlyStop > 0 && bestRound >= 0 && bestRound+1 < len(m.Rounds) {
		m.Rounds = m.Rounds[:bestRound+1]
	}
	return m, nil
}

// valLoss is mean logloss over the held-out indices — the early-stopping metric.
func valLoss(raw [][]float64, y []float64, idx []int, k int) float64 {
	if len(idx) == 0 {
		return 0
	}
	var s float64
	if k == 1 {
		for _, i := range idx {
			p := clamp(sigmoid(raw[i][0]), 1e-15, 1-1e-15)
			if y[i] == 1 {
				s -= math.Log(p)
			} else {
				s -= math.Log(1 - p)
			}
		}
	} else {
		prob := make([]float64, k)
		for _, i := range idx {
			softmax(raw[i], prob)
			s -= math.Log(clamp(prob[int(y[i])], 1e-15, 1))
		}
	}
	return s / float64(len(idx))
}

func clampInt(n, lo, hi int) int {
	if n < lo {
		return lo
	}
	if n > hi {
		return hi
	}
	return n
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

// gradHess is the shared GBDT derivative shape for a probability p and 0/1
// target: gradient p−target, hessian p(1−p) floored to keep splits finite.
func gradHess(p, target float64) (g, h float64) {
	const eps = 1e-6
	return p - target, math.Max(p*(1-p), eps)
}
