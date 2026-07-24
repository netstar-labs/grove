package serve

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/netstar-labs/grove"
)

// trainData builds a small separable 3-class set: class from x0 (and x1).
func trainData(n int) ([][]float64, []float64) {
	rng := rand.New(rand.NewSource(1))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		x := []float64{rng.Float64(), rng.Float64(), rng.Float64()}
		c := 0.0
		if x[0] > 0.66 {
			c = 2
		} else if x[0] > 0.33 {
			c = 1
		}
		X[i], y[i] = x, c
	}
	return X, y
}

// post is a tiny JSON HTTP helper against a handler.
func hreq(t *testing.T, h http.Handler, method, path string, body any) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	r := httptest.NewRequest(method, path, &buf)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	var out map[string]any
	if w.Body.Len() > 0 {
		_ = json.Unmarshal(w.Body.Bytes(), &out)
	}
	return w, out
}

func TestConnectorFlow(t *testing.T) {
	dir := t.TempDir()
	X, y := trainData(1200)

	// 1. receive raw data → train → save (server A)
	a := New(dir).Handler()
	w, out := hreq(t, a, "POST", "/train", TrainRequest{
		Params:   grove.Params{Objective: grove.Multiclass, NumClass: 3, Rounds: 80, MaxDepth: 4},
		Features: X, Labels: y,
		Classes: []string{"low", "mid", "high"},
		Save:    "demo",
	})
	if w.Code != 200 {
		t.Fatalf("train: %d %v", w.Code, out)
	}
	if out["saved"] != "demo" || out["trees"].(float64) == 0 {
		t.Fatalf("train response: %v", out)
	}

	// 2. reload into a FRESH server (proves save/load round-trips through disk)
	b := New(dir).Handler()
	if w, out := hreq(t, b, "POST", "/load?name=demo", nil); w.Code != 200 || out["num_class"].(float64) != 3 {
		t.Fatalf("load: %d %v", w.Code, out)
	}

	// 3. predict on the reloaded model → model response with class labels
	w, _ = hreq(t, b, "POST", "/predict", PredictRequest{Features: [][]float64{
		{0.1, 0.5, 0.5}, // low
		{0.5, 0.5, 0.5}, // mid
		{0.9, 0.5, 0.5}, // high
	}})
	if w.Code != 200 {
		t.Fatalf("predict: %d", w.Code)
	}
	var pr PredictResponse
	_ = json.Unmarshal(w.Body.Bytes(), &pr)
	if len(pr.Labels) != 3 || pr.Labels[0] != "low" || pr.Labels[2] != "high" {
		t.Fatalf("predict labels = %v, want [low mid high]", pr.Labels)
	}

	// 4. GET /model metadata
	if w, out := hreq(t, b, "GET", "/model", nil); w.Code != 200 || out["num_feature"].(float64) != 3 {
		t.Fatalf("model info: %d %v", w.Code, out)
	}
}

func TestConnectorErrors(t *testing.T) {
	h := New(t.TempDir()).Handler()
	// predict with no model loaded → 409
	if w, _ := hreq(t, h, "POST", "/predict", PredictRequest{Features: [][]float64{{1, 2}}}); w.Code != http.StatusConflict {
		t.Errorf("predict-no-model code = %d, want 409", w.Code)
	}
	// load a missing model → 404
	if w, _ := hreq(t, h, "POST", "/load?name=nope", nil); w.Code != http.StatusNotFound {
		t.Errorf("load-missing code = %d, want 404", w.Code)
	}
	// path-traversal model name rejected
	if _, err := New(t.TempDir()).path("../../etc/passwd"); err == nil {
		t.Error("path traversal name should be rejected")
	}
}
