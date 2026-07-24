package main

import (
	"math"
	"testing"
)

// FuzzFeatures fuzzes the CSV cell parser: it must never panic, and must never
// accept a non-numeric or non-finite value (NaN/Inf would route silently).
func FuzzFeatures(f *testing.F) {
	for _, s := range []string{"1.5", "-3", "0", "NaN", "Inf", "-Inf", "1e400", "abc", "", "  4  ", "1,2"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, cell string) {
		x, err := features([]string{cell}, []int{0})
		if err != nil {
			return // rejected — fine
		}
		if len(x) != 1 {
			t.Fatalf("accepted %q but returned %d values", cell, len(x))
		}
		if math.IsNaN(x[0]) || math.IsInf(x[0], 0) {
			t.Fatalf("accepted non-finite value from %q: %v", cell, x[0])
		}
	})
}

// FuzzSplitSet ensures the -ignore parser is total.
func FuzzSplitSet(f *testing.F) {
	f.Add("source,rule_score")
	f.Add(",,,")
	f.Fuzz(func(t *testing.T, s string) {
		_ = splitSet(s)
	})
}
