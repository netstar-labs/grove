package grove

import "testing"

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
