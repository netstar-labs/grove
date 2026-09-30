package mcp

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
)

func req(id int, method string, params any) string {
	m := map[string]any{"jsonrpc": "2.0", "id": id, "method": method}
	if params != nil {
		m["params"] = params
	}
	b, _ := json.Marshal(m)
	return string(b)
}

func TestMCPFlow(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n = 800
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		x := []float64{rng.Float64(), rng.Float64()}
		lbl := 0.0
		if x[0] > 0.5 {
			lbl = 1
		}
		X[i], y[i] = x, lbl
	}

	stream := strings.Join([]string{
		req(1, "initialize", nil),
		req(2, "tools/list", nil),
		req(3, "tools/call", map[string]any{
			"name": "grove_train",
			"arguments": map[string]any{
				"params":   map[string]any{"Objective": "binary", "Rounds": 40, "MaxDepth": 3},
				"features": X, "labels": y, "save": "m",
			},
		}),
		req(4, "tools/call", map[string]any{
			"name":      "grove_predict",
			"arguments": map[string]any{"features": [][]float64{{0.9, 0.5}, {0.1, 0.5}}},
		}),
	}, "\n")

	var out bytes.Buffer
	if err := New(t.TempDir()).Serve(strings.NewReader(stream), &out); err != nil {
		t.Fatal(err)
	}

	dec := json.NewDecoder(&out)
	var resps []rpcResp
	for dec.More() {
		var r rpcResp
		if err := dec.Decode(&r); err != nil {
			t.Fatal(err)
		}
		resps = append(resps, r)
	}
	if len(resps) != 4 {
		t.Fatalf("got %d responses, want 4", len(resps))
	}

	// initialize
	if m, _ := resps[0].Result.(map[string]any); m["protocolVersion"] == nil {
		t.Errorf("initialize missing protocolVersion: %v", resps[0].Result)
	}
	// tools/list — 5 tools
	if m, _ := resps[1].Result.(map[string]any); len(m["tools"].([]any)) != 5 {
		t.Errorf("tools/list = %v", resps[1].Result)
	}
	// train — content text mentions trees
	if txt := toolResultText(t, resps[2]); !strings.Contains(txt, `"trees"`) {
		t.Errorf("train result: %s", txt)
	}
	// predict — high x0 → class 1, low x0 → class 0
	txt := toolResultText(t, resps[3])
	if !strings.Contains(txt, `"classes":[1,0]`) {
		t.Errorf("predict result: %s", txt)
	}
}

// grove_load's own advertised schema is an object {"name": "..."}, same shape
// as grove_save's — a well-formed call following that schema must succeed,
// not fail trying to unmarshal the object into whatever Go type Load's bare
// parameter happens to be.
func TestMCPLoadRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const n = 200
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		x := []float64{rng.Float64()}
		lbl := 0.0
		if x[0] > 0.5 {
			lbl = 1
		}
		X[i], y[i] = x, lbl
	}

	stream := strings.Join([]string{
		req(1, "initialize", nil),
		req(2, "tools/call", map[string]any{
			"name": "grove_train",
			"arguments": map[string]any{
				"params":   map[string]any{"Objective": "binary", "Rounds": 10, "MaxDepth": 2},
				"features": X, "labels": y, "save": "m1",
			},
		}),
		req(3, "tools/call", map[string]any{
			"name":      "grove_load",
			"arguments": map[string]any{"name": "m1"},
		}),
	}, "\n")

	var out bytes.Buffer
	if err := New(t.TempDir()).Serve(strings.NewReader(stream), &out); err != nil {
		t.Fatal(err)
	}

	dec := json.NewDecoder(&out)
	var resps []rpcResp
	for dec.More() {
		var r rpcResp
		if err := dec.Decode(&r); err != nil {
			t.Fatal(err)
		}
		resps = append(resps, r)
	}
	if len(resps) != 3 {
		t.Fatalf("got %d responses, want 3", len(resps))
	}
	if txt := toolResultText(t, resps[2]); !strings.Contains(txt, `"objective"`) {
		t.Errorf("grove_load result missing model info: %s", txt)
	}
}

// withRecover is the only thing standing between a bug anywhere in the
// dispatch chain and the whole stdio session dying — verified directly with
// a synthetic panic (this session has no known live panic trigger left, so
// unlike the other regression tests here, this exercises the containment
// mechanism itself rather than reproducing a specific still-open bug).
func TestWithRecoverContainsAPanic(t *testing.T) {
	resp, notification := withRecover(json.RawMessage(`1`), func() (rpcResp, bool) {
		panic("boom")
	})
	if notification {
		t.Fatal("a recovered panic must still get a response, not be treated as a notification")
	}
	if resp.Error == nil || resp.Error.Code != -32603 {
		t.Fatalf("resp = %+v, want an internal-error (-32603) response", resp)
	}
}

func toolResultText(t *testing.T, r rpcResp) string {
	t.Helper()
	m, ok := r.Result.(map[string]any)
	if !ok {
		t.Fatalf("result not an object: %v", r.Result)
	}
	if m["isError"] == true {
		t.Fatalf("tool returned error: %v", m)
	}
	content := m["content"].([]any)
	return content[0].(map[string]any)["text"].(string)
}
