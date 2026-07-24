package grove

import (
	"bytes"
	"testing"
)

// FuzzFit throws arbitrary small datasets, labels, and shapes at Fit — including
// degenerate ones (single class, constant features, tiny n) — and asserts it
// never panics and that a returned model scores cleanly.
func FuzzFit(f *testing.F) {
	f.Add([]byte{5, 3, 2, 7, 11, 13, 17, 19, 23, 29})
	f.Add([]byte{2, 1, 3, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 4 {
			return
		}
		n := 2 + int(data[0]%24) // 2..25 rows
		d := 1 + int(data[1]%6)  // 1..6 features
		k := 2 + int(data[2]%3)  // 2..4 classes
		X := make([][]float64, n)
		ym := make([]float64, n) // multiclass labels
		yb := make([]float64, n) // binary labels
		bi := 3
		next := func() byte { b := data[bi%len(data)]; bi++; return b }
		for i := range X {
			row := make([]float64, d)
			for j := range row {
				row[j] = float64(int(next())) - 128
			}
			X[i] = row
			ym[i] = float64(int(next()) % k)
			yb[i] = float64(int(next()) % 2)
		}

		if _, err := Fit(X, yb, Params{Objective: Binary, Rounds: 3, MaxDepth: 3, Subsample: 0.8}); err != nil {
			return
		}
		m, err := Fit(X, ym, Params{Objective: Multiclass, NumClass: k, Rounds: 3, MaxDepth: 3})
		if err != nil || m == nil {
			return
		}
		m.PredictClass(X[0])
		m.Predict(X[0])
		m.PredictClassProba(X[0])
		m.Importance()
	})
}

// FuzzLoadPredict is the security-relevant target: Load must reject any
// structurally unsound model, and any model it accepts must be safe to score —
// no panic from bad node links, out-of-range features, or class-count mismatch.
func FuzzLoadPredict(f *testing.F) {
	X, y := genXOR(200, 1)
	bm, _ := Fit(X, y, Params{Objective: Binary, Rounds: 5, MaxDepth: 3})
	var buf bytes.Buffer
	_ = bm.Save(&buf)
	f.Add(buf.Bytes())
	f.Add([]byte(`{"num_class":1,"num_feature":2,"base":[0],"rounds":[[{"nodes":[{"f":0,"t":0.5,"l":1,"r":2},{"leaf":true,"v":1},{"leaf":true,"v":-1}]}]]}`))
	f.Add([]byte(`{"num_class":2,"num_feature":1,"base":[0,0],"rounds":[[{"nodes":[{"leaf":true}]},{"nodes":[{"leaf":true}]}]]}`))
	f.Add([]byte(`{"num_class":1,"num_feature":1,"base":[0],"rounds":[[{"nodes":[{"f":9,"l":1,"r":1}]}]]}`)) // bad links/feature

	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := Load(bytes.NewReader(data))
		if err != nil {
			return // rejected — the point of validate()
		}
		if m.NumFeature > 4096 {
			return // avoid a huge harness allocation on an absurd (but valid) header
		}
		x := make([]float64, m.NumFeature)
		// none of these may panic on an accepted model
		m.PredictClass(x)
		m.PredictClassProba(x)
		m.Predict(x)
		m.Importance()
		m.PredictClass(nil) // short vector must be handled, not crash
	})
}
