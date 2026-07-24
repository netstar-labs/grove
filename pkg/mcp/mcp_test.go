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
