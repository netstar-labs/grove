package grove

import (
	"math/rand"
	"testing"
)

// genWide is a wide-feature binary problem (label from x0,x1; the rest noise) —
// the shape where feature-parallel split finding pays off.
func genWide(n, d int, seed int64) ([][]float64, []float64) {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		x := make([]float64, d)
		for j := range x {
			x[j] = rng.Float64()
		}
		lbl := 0.0
		if (x[0] > 0.5) != (x[1] > 0.5) {
			lbl = 1
		}
		X[i], y[i] = x, lbl
	}
	return X, y
}

// BenchmarkFitWide exercises the parallel path; compare -cpu 1 (serial) vs
// -cpu 10 to see the split-finding speedup: go test -bench FitWide -cpu 1,10.
func BenchmarkFitWide(b *testing.B) {
	X, y := genWide(5000, 40, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Fit(X, y, Params{Objective: Binary, Rounds: 60, MaxDepth: 6, LearningRate: 0.1}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFitBinary(b *testing.B) {
	X, y := genXOR(5000, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Fit(X, y, Params{Objective: Binary, Rounds: 100, MaxDepth: 5, LearningRate: 0.1}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFitMulticlass(b *testing.B) {
	X, y := gen3(5000, 101)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Fit(X, y, Params{Objective: Multiclass, NumClass: 3, Rounds: 60, MaxDepth: 5}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPredict(b *testing.B) {
	X, y := genXOR(5000, 102)
	m, _ := Fit(X, y, Params{Objective: Binary, Rounds: 100, MaxDepth: 5})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.PredictClass(X[i%len(X)])
	}
}
