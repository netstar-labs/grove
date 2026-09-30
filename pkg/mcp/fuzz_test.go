package mcp

import (
	"bytes"
	"testing"
)

// FuzzServe feeds arbitrary bytes into the JSON-RPC stdio loop end to end —
// decode, method dispatch, and every tool arm (including grove_train, the
// entry point for two of this pass's three confirmed bugs). The only
// invariant: it must never panic (withRecover notwithstanding — a fuzz
// target should never rely on the recovery net to call itself passing) and
// must never hang; a malformed or short frame returning an error from Serve
// is expected and fine.
func FuzzServe(f *testing.F) {
	f.Add([]byte(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
	f.Add([]byte(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}`))
	f.Add([]byte(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"grove_train","arguments":{"params":{"Objective":"binary","Rounds":2},"features":[[1],[2]],"labels":[0,1]}}}`))
	f.Add([]byte(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"grove_load","arguments":{"name":"m1"}}}`))
	f.Add([]byte(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"grove_predict","arguments":{"features":[[1]]}}}`))
	f.Add([]byte(`{"jsonrpc":"2.0","method":"notifications/initialized"}`))
	f.Add([]byte(`not json at all`))
	f.Add([]byte(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":`)) // truncated
	f.Fuzz(func(t *testing.T, data []byte) {
		var out bytes.Buffer
		_ = New(t.TempDir()).Serve(bytes.NewReader(data), &out)
	})
}
