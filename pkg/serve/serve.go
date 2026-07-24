// Package serve exposes grove over a network: receive raw feature data, train a
// model, save/load it to a model directory, and score rows — returning a model
// response. It holds one current model, swapped atomically under a lock. The
// same core methods back both the HTTP handler (also mountable on a unix socket)
// and the MCP server, so the three transports share one implementation.
package serve

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/netstar-labs/grove"
)

// Server owns the current model and the directory saved models live in.
type Server struct {
	mu    sync.RWMutex
	model *grove.Model
	dir   string
}

// New returns a server that saves/loads models under dir (created if absent).
func New(dir string) *Server { return &Server{dir: dir} }

// ---- wire types -------------------------------------------------------------

// TrainRequest carries the raw training matrix and hyperparameters.
type TrainRequest struct {
	Params       grove.Params `json:"params"`   // objective, rounds, depth, …
	Features     [][]float64  `json:"features"` // n×d row-major
	Labels       []float64    `json:"labels"`   // n: 0/1, class index, or continuous
	FeatureNames []string     `json:"feature_names,omitempty"`
	Classes      []string     `json:"classes,omitempty"` // class-index → label
	Save         string       `json:"save,omitempty"`    // if set, also persist under this name
}

// TrainResponse summarizes the freshly trained (and now-current) model.
type TrainResponse struct {
	Objective  string    `json:"objective"`
	NumClass   int       `json:"num_class"`
	Trees      int       `json:"trees"`
	Classes    []string  `json:"classes,omitempty"`
	Importance []float64 `json:"importance"`
	Saved      string    `json:"saved,omitempty"`
}

// PredictRequest is a batch of feature rows to score with the current model.
type PredictRequest struct {
	Features [][]float64 `json:"features"`
}

// PredictResponse holds classification or regression outputs (whichever fits
// the current model's objective).
type PredictResponse struct {
	Classes       []int       `json:"classes,omitempty"`       // predicted class index
	Labels        []string    `json:"labels,omitempty"`        // predicted class label (if named)
	Probabilities [][]float64 `json:"probabilities,omitempty"` // full distribution
	Values        []float64   `json:"values,omitempty"`        // regression predictions
}

// ModelInfo describes the current (or just-loaded) model.
type ModelInfo struct {
	Objective    string    `json:"objective"`
	NumClass     int       `json:"num_class"`
	NumFeature   int       `json:"num_feature"`
	Trees        int       `json:"trees"`
	Classes      []string  `json:"classes,omitempty"`
	FeatureNames []string  `json:"feature_names,omitempty"`
	Importance   []float64 `json:"importance"`
}

var errNoModel = errors.New("serve: no model loaded")

// ---- core methods (transport-independent) -----------------------------------

// Train fits a model from the request, makes it current, and optionally saves it.
func (s *Server) Train(req TrainRequest) (TrainResponse, error) {
	m, err := grove.Fit(req.Features, req.Labels, req.Params)
	if err != nil {
		return TrainResponse{}, err
	}
	m.FeatureNames = req.FeatureNames
	m.Classes = req.Classes

	s.mu.Lock()
	s.model = m
	s.mu.Unlock()

	resp := TrainResponse{
		Objective: m.Objective, NumClass: m.NumClass, Trees: m.TreeCount(),
		Classes: m.Classes, Importance: m.Importance(),
	}
	if req.Save != "" {
		if err := s.persist(req.Save, m); err != nil {
			return resp, err
		}
		resp.Saved = req.Save
	}
	return resp, nil
}

// Predict scores rows with the current model.
func (s *Server) Predict(req PredictRequest) (PredictResponse, error) {
	m := s.current()
	if m == nil {
		return PredictResponse{}, errNoModel
	}
	var resp PredictResponse
	if m.Objective == grove.Regression {
		resp.Values = make([]float64, len(req.Features))
		for i, x := range req.Features {
			resp.Values[i] = m.PredictValue(x)
		}
		return resp, nil
	}
	resp.Classes = make([]int, len(req.Features))
	resp.Probabilities = make([][]float64, len(req.Features))
	named := len(m.Classes) > 0
	if named {
		resp.Labels = make([]string, len(req.Features))
	}
	for i, x := range req.Features {
		c := m.PredictClass(x)
		resp.Classes[i] = c
		resp.Probabilities[i] = m.Predict(x)
		if named && c >= 0 && c < len(m.Classes) {
			resp.Labels[i] = m.Classes[c]
		}
	}
	return resp, nil
}

// Save persists the current model under name.
func (s *Server) Save(name string) error {
	m := s.current()
	if m == nil {
		return errNoModel
	}
	return s.persist(name, m)
}

// Load reads a saved model and makes it current.
func (s *Server) Load(name string) (ModelInfo, error) {
	path, err := s.path(name)
	if err != nil {
		return ModelInfo{}, err
	}
	f, err := os.Open(path)
	if err != nil {
		return ModelInfo{}, err
	}
	defer f.Close()
	m, err := grove.Load(f)
	if err != nil {
		return ModelInfo{}, err
	}
	s.mu.Lock()
	s.model = m
	s.mu.Unlock()
	return info(m), nil
}

// Info returns metadata about the current model.
func (s *Server) Info() (ModelInfo, error) {
	m := s.current()
	if m == nil {
		return ModelInfo{}, errNoModel
	}
	return info(m), nil
}

func (s *Server) current() *grove.Model {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.model
}

func info(m *grove.Model) ModelInfo {
	return ModelInfo{
		Objective: m.Objective, NumClass: m.NumClass, NumFeature: m.NumFeature,
		Trees: m.TreeCount(), Classes: m.Classes, FeatureNames: m.FeatureNames,
		Importance: m.Importance(),
	}
}

// persist writes m to dir/<safe name>.json.
func (s *Server) persist(name string, m *grove.Model) error {
	path, err := s.path(name)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return m.Save(f)
}

// path resolves a model name to a file under dir, rejecting anything that isn't
// a plain base name (no separators, no traversal).
func (s *Server) path(name string) (string, error) {
	if name == "" || name != filepath.Base(name) || name == "." || name == ".." {
		return "", fmt.Errorf("serve: invalid model name %q", name)
	}
	for _, r := range name {
		if !(r == '-' || r == '_' || r == '.' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')) {
			return "", fmt.Errorf("serve: invalid model name %q", name)
		}
	}
	return filepath.Join(s.dir, name+".json"), nil
}

// ---- HTTP handler (also serves over a unix socket) --------------------------

// Handler returns the HTTP mux. Mount it on a TCP or unix listener.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /train", s.hTrain)
	mux.HandleFunc("POST /predict", s.hPredict)
	mux.HandleFunc("POST /save", s.hSave)
	mux.HandleFunc("POST /load", s.hLoad)
	mux.HandleFunc("GET /model", s.hInfo)
	return mux
}

func (s *Server) hTrain(w http.ResponseWriter, r *http.Request) {
	var req TrainRequest
	if !decode(w, r, &req) {
		return
	}
	resp, err := s.Train(req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, resp)
}

func (s *Server) hPredict(w http.ResponseWriter, r *http.Request) {
	var req PredictRequest
	if !decode(w, r, &req) {
		return
	}
	resp, err := s.Predict(req)
	if err != nil {
		writeErr(w, http.StatusConflict, err)
		return
	}
	writeJSON(w, resp)
}

func (s *Server) hSave(w http.ResponseWriter, r *http.Request) {
	if err := s.Save(r.URL.Query().Get("name")); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, map[string]string{"saved": r.URL.Query().Get("name")})
}

func (s *Server) hLoad(w http.ResponseWriter, r *http.Request) {
	mi, err := s.Load(r.URL.Query().Get("name"))
	if err != nil {
		writeErr(w, http.StatusNotFound, err)
		return
	}
	writeJSON(w, mi)
}

func (s *Server) hInfo(w http.ResponseWriter, r *http.Request) {
	mi, err := s.Info()
	if err != nil {
		writeErr(w, http.StatusConflict, err)
		return
	}
	writeJSON(w, mi)
}

const maxBody = 256 << 20 // 256 MiB training payload cap

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody))
	if err := dec.Decode(v); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
}
