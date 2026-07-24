// Package grove is a small, dependency-free gradient-boosted decision tree
// library: it grows an ensemble of shallow regression trees over histogram-
// binned features to model binary (logistic) and multiclass (softmax) targets,
// then scores new rows with a pure-Go Predict and reports which features drove
// the fit. Training and inference share one toolchain and zero third-party deps,
// so a model trains offline and scores inline anywhere Go runs — no Python
// runtime, no native library, a static binary on any host.
//
// The whole engine is standard library only. The design favours legibility and
// determinism over raw speed: a seeded fit is reproducible, and every tree is a
// plain slice of nodes you can serialize, inspect, and diff.
package grove

import (
	"errors"
	"math"
	"math/rand"
)

// Objective names the loss the ensemble is trained against.
const (
	Binary     = "binary"     // logistic loss over {0,1} labels
	Multiclass = "multiclass" // softmax over class indices [0, NumClass)
)

// Params configures a fit. The zero value is not valid; use Default and adjust,
// or rely on Fit filling unset (zero) fields with the defaults below.
type Params struct {
	Objective      string  // Binary or Multiclass
	NumClass       int     // classes for Multiclass (ignored for Binary)
	Rounds         int     // boosting rounds (trees for Binary, Rounds×NumClass total)
	LearningRate   float64 // shrinkage applied to each tree's leaves
	MaxDepth       int     // maximum tree depth
	MaxBins        int     // histogram bins per feature (2..256)
	Lambda         float64 // L2 regularization on leaf weights
	Gamma          float64 // minimum gain to make a split
	MinChildWeight float64 // minimum summed hessian in a child
	Subsample      float64 // row fraction sampled per tree (0..1]
	Seed           int64   // RNG seed for subsampling (reproducible)
}

// Default returns sensible starting parameters for the given objective.
func Default(objective string, numClass int) Params {
	return Params{
		Objective:      objective,
		NumClass:       numClass,
		Rounds:         100,
		LearningRate:   0.1,
		MaxDepth:       6,
		MaxBins:        256,
		Lambda:         1,
		Gamma:          0,
		MinChildWeight: 1,
		Subsample:      1,
		Seed:           1,
	}
}

// fill replaces zero-valued fields with defaults and validates the essentials.
func (p *Params) fill() error {
	if p.Objective == "" {
		p.Objective = Binary
	}
	if p.Objective != Binary && p.Objective != Multiclass {
		return errors.New("grove: unknown objective " + p.Objective)
	}
	if p.Objective == Multiclass && p.NumClass < 2 {
		return errors.New("grove: multiclass requires NumClass >= 2")
	}
	d := Default(p.Objective, p.NumClass)
	if p.Rounds <= 0 {
		p.Rounds = d.Rounds
	}
	if p.LearningRate <= 0 {
		p.LearningRate = d.LearningRate
	}
	if p.MaxDepth <= 0 {
		p.MaxDepth = d.MaxDepth
	}
	if p.MaxBins < 2 {
		p.MaxBins = d.MaxBins
	}
	if p.MaxBins > 256 {
		p.MaxBins = 256
	}
	if p.Lambda < 0 {
		p.Lambda = 0
	}
	if p.MinChildWeight <= 0 {
		p.MinChildWeight = d.MinChildWeight
	}
	if p.Subsample <= 0 || p.Subsample > 1 {
		p.Subsample = 1
	}
	if p.Seed == 0 {
		p.Seed = d.Seed
	}
	return nil
}

// ---- small numeric helpers -------------------------------------------------

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// softmax fills out with the normalized exponentials of raw (stable).
func softmax(raw, out []float64) {
	m := raw[0]
	for _, v := range raw[1:] {
		if v > m {
			m = v
		}
	}
	var s float64
	for k, v := range raw {
		e := math.Exp(v - m)
		out[k] = e
		s += e
	}
	for k := range out {
		out[k] /= s
	}
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func iota0(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}

// subsample returns a fresh index slice sampling frac of idx without replacement.
func subsample(idx []int, frac float64, rng *rand.Rand) []int {
	if frac >= 1 {
		return idx
	}
	k := int(frac * float64(len(idx)))
	if k < 1 {
		k = 1
	}
	out := append([]int(nil), idx...)
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out[:k]
}
