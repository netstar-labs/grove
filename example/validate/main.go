// Validation harness: train grove to recover a known decision function and
// confirm it generalizes. The "known vector" here is the labelling rule itself —
// a deterministic 3-class boundary — so a correct learner should reach near-100%
// on held-out data. Exits non-zero if it doesn't, so this doubles as a gate.
//
//	GOWORK=off go run ./example/validate
package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/netstar-labs/grove"
)

// label is the known target function: a piecewise-constant 3-class boundary with
// an x1 interaction. No noise — a correct model should recover it almost exactly.
func label(x []float64) float64 {
	c := 0.0
	switch {
	case x[0] > 0.66:
		c = 2
	case x[0] > 0.33:
		c = 1
	}
	if x[1] > 0.75 {
		c = float64((int(c) + 1) % 3)
	}
	return c
}

func gen(n int, seed int64) ([][]float64, []float64) {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		x := []float64{rng.Float64(), rng.Float64(), rng.Float64(), rng.Float64()}
		X[i], y[i] = x, label(x)
	}
	return X, y
}

func main() {
	Xtr, ytr := gen(5000, 1)
	Xte, yte := gen(2000, 2)

	m, err := grove.Fit(Xtr, ytr, grove.Params{
		Objective: grove.Multiclass, NumClass: 3, Rounds: 150, MaxDepth: 5, LearningRate: 0.15,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	acc := accuracy(m, Xte, yte)
	fmt.Printf("recovered known 3-class boundary: test accuracy = %.4f over %d rows, %d trees\n",
		acc, len(Xte), m.TreeCount())
	fmt.Printf("feature importance (gain): %v  (x0,x1 carry the signal; x2,x3 are noise)\n", m.Importance())

	const threshold = 0.95
	if acc < threshold {
		fmt.Fprintf(os.Stderr, "FAIL: accuracy %.4f below %.2f\n", acc, threshold)
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func accuracy(m *grove.Model, X [][]float64, y []float64) float64 {
	correct := 0
	for i := range X {
		if float64(m.PredictClass(X[i])) == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(X))
}
