package main

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/netstar-labs/grove"
)

// A model's NumClass and len(Classes) are not the same invariant — nothing
// enforces that a stored model's Classes list actually has NumClass entries
// (e.g. trained/loaded over the network via pkg/serve.Train, which sets
// m.Classes = req.Classes with no length check against NumClass). eval()
// must not assume a predicted class index is always a valid index into
// classes just because Classes is non-empty.
func TestEvalHandlesShortClassesList(t *testing.T) {
	X := make([][]float64, 0, 90)
	y := make([]float64, 0, 90)
	for _, v := range []float64{0, 1, 2} {
		for i := 0; i < 30; i++ {
			X = append(X, []float64{v + float64(i%3)*0.01})
			y = append(y, v)
		}
	}
	m, err := grove.Fit(X, y, grove.Params{Objective: grove.Multiclass, NumClass: 3, Rounds: 60, MaxDepth: 4})
	if err != nil {
		t.Fatal(err)
	}
	if cls := m.PredictClass([]float64{2}); cls != 2 {
		t.Fatalf("setup: expected class 2 for x=2, got %d", cls)
	}
	m.FeatureNames = []string{"x"}
	// Deliberately short: NumClass is 3, but only 2 names stored — the exact
	// mismatch that made pIdx (== predIdx == 2) run off the end of the
	// nC == len(classes) == 2 sized tp/fp/fn/support slices.
	m.Classes = []string{"cat", "dog"}

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.json")
	if err := m.SaveFile(modelPath); err != nil {
		t.Fatal(err)
	}

	csvPath := filepath.Join(dir, "eval.csv")
	// x=2 predicts class 2 (verified above), but the actual label given here
	// is "cat" (class 0, present in the truncated Classes) — a genuine
	// misclassification landing on the class index that isn't in Classes.
	csv := "x,label\n" + strconv.Itoa(2) + ",cat\n0,cat\n1,dog\n"
	if err := os.WriteFile(csvPath, []byte(csv), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := eval([]string{"-model", modelPath, "-in", csvPath, "-target", "label"}); err != nil {
		t.Fatalf("eval returned an error (want a clean report, not a panic or error): %v", err)
	}
}
