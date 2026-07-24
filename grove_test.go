package grove

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// genXOR builds a nonlinear binary problem: label = (x0>.5) XOR (x1>.5), with
// four noise features and 5% label noise. A linear model can't separate it; a
// depth-2+ tree ensemble can.
func genXOR(n int, seed int64) ([][]float64, []float64) {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x := make([]float64, 6)
		for f := range x {
			x[f] = rng.Float64()
		}
		lbl := 0.0
		if (x[0] > 0.5) != (x[1] > 0.5) {
			lbl = 1
		}
		if rng.Float64() < 0.05 {
			lbl = 1 - lbl
		}
		X[i], y[i] = x, lbl
	}
	return X, y
}

// gen3 builds a 3-class problem: the class is a piecewise function of x0 with an
// x1 interaction, plus noise features.
func gen3(n int, seed int64) ([][]float64, []float64) {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x := make([]float64, 5)
		for f := range x {
			x[f] = rng.Float64()
		}
		var c float64
		switch {
		case x[0] > 0.66:
			c = 2
		case x[0] > 0.33:
			c = 1
		default:
			c = 0
		}
		if x[1] > 0.8 { // interaction bump
			c = math.Mod(c+1, 3)
		}
		X[i], y[i] = x, c
	}
	return X, y
}

func accuracy(m *Model, X [][]float64, y []float64) float64 {
	correct := 0
	for i := range X {
		if float64(m.PredictClass(X[i])) == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(X))
}

func binaryLogLoss(m *Model, X [][]float64, y []float64) float64 {
	var s float64
	for i := range X {
		p := clamp(m.Predict(X[i])[0], 1e-15, 1-1e-15)
		if y[i] == 1 {
			s -= math.Log(p)
		} else {
			s -= math.Log(1 - p)
		}
	}
	return s / float64(len(X))
}

func TestBinaryLearnsXOR(t *testing.T) {
	Xtr, ytr := genXOR(4000, 1)
	Xte, yte := genXOR(1000, 2)
	m, err := Fit(Xtr, ytr, Params{Objective: Binary, Rounds: 120, MaxDepth: 4, LearningRate: 0.2})
	if err != nil {
		t.Fatal(err)
	}
	acc := accuracy(m, Xte, yte)
	if acc < 0.9 {
		t.Errorf("XOR test accuracy = %.3f, want >= 0.90", acc)
	}
	t.Logf("XOR test accuracy = %.3f, trees = %d", acc, m.TreeCount())
}

func TestMulticlassLearns(t *testing.T) {
	Xtr, ytr := gen3(4000, 3)
	Xte, yte := gen3(1000, 4)
	m, err := Fit(Xtr, ytr, Params{Objective: Multiclass, NumClass: 3, Rounds: 120, MaxDepth: 4, LearningRate: 0.2})
	if err != nil {
		t.Fatal(err)
	}
	acc := accuracy(m, Xte, yte)
	if acc < 0.9 {
		t.Errorf("3-class test accuracy = %.3f, want >= 0.90", acc)
	}
	t.Logf("3-class test accuracy = %.3f", acc)
}

// TestBoostingReducesLoss confirms more rounds cut training loss — boosting works.
func TestBoostingReducesLoss(t *testing.T) {
	X, y := genXOR(3000, 5)
	shallow, _ := Fit(X, y, Params{Objective: Binary, Rounds: 5, MaxDepth: 4, LearningRate: 0.2})
	deep, _ := Fit(X, y, Params{Objective: Binary, Rounds: 150, MaxDepth: 4, LearningRate: 0.2})
	l5, l150 := binaryLogLoss(shallow, X, y), binaryLogLoss(deep, X, y)
	if l150 >= l5 {
		t.Errorf("training loss did not fall: 5 rounds=%.4f, 150 rounds=%.4f", l5, l150)
	}
	t.Logf("train logloss: 5 rounds=%.4f -> 150 rounds=%.4f", l5, l150)
}

// TestImportanceFindsSignal checks the informative features (x0,x1) dominate the
// noise features in the gain ranking.
func TestImportanceFindsSignal(t *testing.T) {
	X, y := genXOR(4000, 6)
	m, _ := Fit(X, y, Params{Objective: Binary, Rounds: 120, MaxDepth: 4, LearningRate: 0.2})
	imp := m.Importance()
	signal := imp[0] + imp[1]
	var noise float64
	for f := 2; f < len(imp); f++ {
		noise += imp[f]
	}
	if signal <= noise {
		t.Errorf("signal importance %.2f should exceed noise %.2f (imp=%v)", signal, noise, imp)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	X, y := genXOR(1500, 7)
	m, _ := Fit(X, y, Params{Objective: Binary, Rounds: 40, MaxDepth: 3})
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	m2, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		if a, b := m.Predict(X[i])[0], m2.Predict(X[i])[0]; math.Abs(a-b) > 1e-12 {
			t.Fatalf("row %d: reloaded prediction %.9f != %.9f", i, b, a)
		}
	}
}

func TestDeterministicSeed(t *testing.T) {
	X, y := genXOR(2000, 8)
	p := Params{Objective: Binary, Rounds: 50, MaxDepth: 4, Subsample: 0.7, Seed: 42}
	m1, _ := Fit(X, y, p)
	m2, _ := Fit(X, y, p)
	for i := range X {
		if m1.Predict(X[i])[0] != m2.Predict(X[i])[0] {
			t.Fatalf("same seed produced different predictions at row %d", i)
		}
	}
}

func TestFitErrors(t *testing.T) {
	if _, err := Fit(nil, nil, Params{}); err == nil {
		t.Error("empty dataset should error")
	}
	if _, err := Fit([][]float64{{1}}, []float64{0, 1}, Params{}); err == nil {
		t.Error("mismatched X/y lengths should error")
	}
	if _, err := Fit([][]float64{{1}}, []float64{5}, Params{Objective: Multiclass, NumClass: 3}); err == nil {
		t.Error("label out of class range should error")
	}
}
