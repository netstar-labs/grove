// Package mcp is a minimal Model Context Protocol stdio server exposing grove to
// AI agents: they can train a model from raw features, save/load it, and score
// rows through the tools grove_train / grove_predict / grove_save / grove_load /
// grove_model_info. It speaks newline-delimited JSON-RPC 2.0 over stdin/stdout
// and is backed by the same serve.Server core as the HTTP/unix transport.
package mcp

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/netstar-labs/grove/pkg/serve"
)

// Server wraps the shared serve core.
type Server struct{ core *serve.Server }

// New returns an MCP server saving/loading models under dir.
func New(dir string) *Server { return &Server{core: serve.New(dir)} }

type rpcReq struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResp struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcErr         `json:"error,omitempty"`
}

type rpcErr struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Serve runs the JSON-RPC loop until in is exhausted.
func (s *Server) Serve(in io.Reader, out io.Writer) error {
	dec := json.NewDecoder(in)
	enc := json.NewEncoder(out)
	for {
		var req rpcReq
		if err := dec.Decode(&req); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		resp, notification := s.handle(&req)
		if notification {
			continue // notifications carry no id and get no response
		}
		if err := enc.Encode(resp); err != nil {
			return err
		}
	}
}

func (s *Server) handle(req *rpcReq) (rpcResp, bool) {
	switch req.Method {
	case "initialize":
		return result(req.ID, map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": "grove", "version": "0.1"},
		}), false
	case "notifications/initialized":
		return rpcResp{}, true
	case "ping":
		return result(req.ID, map[string]any{}), false
	case "tools/list":
		return result(req.ID, map[string]any{"tools": tools}), false
	case "tools/call":
		return s.callTool(req), false
	default:
		return fail(req.ID, -32601, "method not found: "+req.Method), false
	}
}

func (s *Server) callTool(req *rpcReq) rpcResp {
	var call struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(req.Params, &call); err != nil {
		return fail(req.ID, -32602, "bad params: "+err.Error())
	}

	var payload any
	var err error
	switch call.Name {
	case "grove_train":
		var tr serve.TrainRequest
		if err = json.Unmarshal(call.Arguments, &tr); err == nil {
			payload, err = s.core.Train(tr)
		}
	case "grove_predict":
		var pr serve.PredictRequest
		if err = json.Unmarshal(call.Arguments, &pr); err == nil {
			payload, err = s.core.Predict(pr)
		}
	case "grove_save":
		var a struct {
			Name string `json:"name"`
		}
		if err = json.Unmarshal(call.Arguments, &a); err == nil {
			if err = s.core.Save(a.Name); err == nil {
				payload = map[string]string{"saved": a.Name}
			}
		}
	case "grove_load":
		var a struct {
			Name string `json:"name"`
		}
		if err = json.Unmarshal(call.Arguments, &a); err == nil {
			payload, err = s.core.Load(a.Name)
		}
	case "grove_model_info":
		payload, err = s.core.Info()
	default:
		return fail(req.ID, -32602, "unknown tool: "+call.Name)
	}

	if err != nil {
		return result(req.ID, toolText(fmt.Sprintf("error: %v", err), true))
	}
	body, _ := json.Marshal(payload)
	return result(req.ID, toolText(string(body), false))
}

// toolText wraps a string as an MCP tool result content block.
func toolText(text string, isError bool) map[string]any {
	return map[string]any{
		"content": []map[string]any{{"type": "text", "text": text}},
		"isError": isError,
	}
}

func result(id json.RawMessage, r any) rpcResp {
	return rpcResp{JSONRPC: "2.0", ID: id, Result: r}
}

func fail(id json.RawMessage, code int, msg string) rpcResp {
	return rpcResp{JSONRPC: "2.0", ID: id, Error: &rpcErr{Code: code, Message: msg}}
}

// tools is the advertised tool set. Schemas are intentionally light: features is
// an array of numeric rows; params mirrors grove.Params.
var tools = []map[string]any{
	{
		"name":        "grove_train",
		"description": "Train a grove model from raw feature rows and labels; makes it current and optionally saves it.",
		"inputSchema": obj(map[string]any{
			"params":        obj(nil),
			"features":      arr(arr(num())),
			"labels":        arr(num()),
			"feature_names": arr(str()),
			"classes":       arr(str()),
			"save":          str(),
		}, "features", "labels"),
	},
	{
		"name":        "grove_predict",
		"description": "Score feature rows with the current model; returns class labels/probabilities or regression values.",
		"inputSchema": obj(map[string]any{"features": arr(arr(num()))}, "features"),
	},
	{
		"name":        "grove_save",
		"description": "Persist the current model under a name.",
		"inputSchema": obj(map[string]any{"name": str()}, "name"),
	},
	{
		"name":        "grove_load",
		"description": "Load a saved model by name and make it current.",
		"inputSchema": obj(map[string]any{"name": str()}, "name"),
	},
	{
		"name":        "grove_model_info",
		"description": "Report the current model's objective, classes, tree count, and feature importance.",
		"inputSchema": obj(map[string]any{}),
	},
}

func obj(props map[string]any, required ...string) map[string]any {
	m := map[string]any{"type": "object"}
	if props != nil {
		m["properties"] = props
	}
	if len(required) > 0 {
		m["required"] = required
	}
	return m
}
func arr(items any) map[string]any { return map[string]any{"type": "array", "items": items} }
func num() map[string]any          { return map[string]any{"type": "number"} }
func str() map[string]any          { return map[string]any{"type": "string"} }
