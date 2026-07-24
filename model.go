package grove

import (
	"encoding/json"
	"io"
)

// Model is a trained ensemble: a base score per class plus a list of boosting
// rounds, each holding one tree per class (one tree per round for Binary). It is
// pure data — safe to serialize, embed, and score without the training code.
type Model struct {
	Objective    string    `json:"objective"`
	NumClass     int       `json:"num_class"` // 1 for Binary
	NumFeature   int       `json:"num_feature"`
	Base         []float64 `json:"base"`
	LearningRate float64   `json:"learning_rate"`
	Rounds       [][]tree  `json:"rounds"`
	FeatureNames []string  `json:"feature_names,omitempty"` // column order Predict expects
	Classes      []string  `json:"classes,omitempty"`       // label for each class index (optional)
}

// rawScores returns the accumulated per-class score (base + every tree).
func (m *Model) rawScores(x []float64) []float64 {
	out := append([]float64(nil), m.Base...)
	for _, round := range m.Rounds {
		for c := range round {
			out[c] += round[c].predict(x)
		}
	}
	return out
}

// Predict returns calibrated probabilities: a single P(class=1) for Binary, or a
// NumClass-length distribution for Multiclass.
func (m *Model) Predict(x []float64) []float64 {
	raw := m.rawScores(x)
	if m.NumClass == 1 {
		return []float64{sigmoid(raw[0])}
	}
	out := make([]float64, m.NumClass)
	softmax(raw, out)
	return out
}

// PredictClassProba returns the most likely class and its probability in a
// single pass over the ensemble — cheaper than calling PredictClass and Predict
// separately (each of which walks every tree).
func (m *Model) PredictClassProba(x []float64) (int, float64) {
	raw := m.rawScores(x)
	if m.NumClass == 1 {
		p := sigmoid(raw[0])
		if p >= 0.5 {
			return 1, p
		}
		return 0, 1 - p
	}
	prob := make([]float64, m.NumClass)
	softmax(raw, prob)
	best := 0
	for c := 1; c < len(prob); c++ {
		if prob[c] > prob[best] {
			best = c
		}
	}
	return best, prob[best]
}

// PredictClass returns the most likely class index (0/1 for Binary).
func (m *Model) PredictClass(x []float64) int {
	raw := m.rawScores(x)
	if m.NumClass == 1 {
		if sigmoid(raw[0]) >= 0.5 {
			return 1
		}
		return 0
	}
	best := 0
	for c := 1; c < len(raw); c++ {
		if raw[c] > raw[best] {
			best = c
		}
	}
	return best
}

// Importance returns each feature's total split gain across the ensemble — a
// global, interpretable ranking of what the model actually used.
func (m *Model) Importance() []float64 {
	imp := make([]float64, m.NumFeature)
	for _, round := range m.Rounds {
		for c := range round {
			for _, nd := range round[c].Nodes {
				if !nd.Leaf && nd.Feature < len(imp) {
					imp[nd.Feature] += nd.Gain
				}
			}
		}
	}
	return imp
}

// TreeCount reports the total number of trees in the ensemble.
func (m *Model) TreeCount() int { return len(m.Rounds) * m.NumClass }

// Save writes the model as indented JSON, so a retrain produces a readable
// git diff rather than one changed line.
func (m *Model) Save(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", " ")
	return enc.Encode(m)
}

// Load reads a model written by Save.
func Load(r io.Reader) (*Model, error) {
	var m Model
	if err := json.NewDecoder(r).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}
