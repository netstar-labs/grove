// Embed demo: train a grove model in-process and score with it — the library
// used directly, no CLI, no files.
//
//	GOWORK=off go run ./example/embed
package main

import (
	"fmt"
	"math/rand"

	"github.com/netstar-labs/grove"
)

func main() {
	// A nonlinear binary problem: label = (x0>.5) XOR (x1>.5); x2 is noise.
	rng := rand.New(rand.NewSource(1))
	const n = 3000
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		x := []float64{rng.Float64(), rng.Float64(), rng.Float64()}
		lbl := 0.0
		if (x[0] > 0.5) != (x[1] > 0.5) {
			lbl = 1
		}
		X[i], y[i] = x, lbl
	}

	m, err := grove.Fit(X, y, grove.Params{
		Objective: grove.Binary, Rounds: 100, MaxDepth: 4, LearningRate: 0.2,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("trained %d trees\n", m.TreeCount())
	fmt.Printf("P(1 | x0=.2 x1=.8) = %.3f  (XOR true  -> ~1)\n", m.Predict([]float64{0.2, 0.8, 0.5})[0])
	fmt.Printf("P(1 | x0=.2 x1=.2) = %.3f  (XOR false -> ~0)\n", m.Predict([]float64{0.2, 0.2, 0.5})[0])
	fmt.Printf("feature importance (gain) = %v  (x0,x1 dominate x2)\n", m.Importance())
}
