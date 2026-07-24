package grove

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// modelVersion is the on-disk model schema version. Load rejects anything newer
// so a future format can't be silently mis-read by an older binary.
const modelVersion = 1

// Model is a trained ensemble: a base score per class plus a list of boosting
// rounds, each holding one tree per class (one tree per round for Binary). It is
// pure data — safe to serialize, embed, and score without the training code.
type Model struct {
	Version      int       `json:"version"`
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

// rawBinary is the allocation-free score path for a Binary model — the common
// case for inline scoring.
func (m *Model) rawBinary(x []float64) float64 {
	s := m.Base[0]
	for _, round := range m.Rounds {
		s += round[0].predict(x)
	}
	return s
}

// Predict returns calibrated probabilities: a single P(class=1) for Binary, or a
// NumClass-length distribution for Multiclass. A feature vector shorter than
// NumFeature yields a neutral prediction rather than a panic.
func (m *Model) Predict(x []float64) []float64 {
	if len(x) < m.NumFeature {
		return neutral(m.NumClass)
	}
	if m.NumClass == 1 {
		return []float64{sigmoid(m.rawBinary(x))}
	}
	out := make([]float64, m.NumClass)
	softmax(m.rawScores(x), out)
	return out
}

// neutral is the no-information prediction used when a caller passes too few
// features: 0.5 for Binary, a uniform distribution for Multiclass.
func neutral(k int) []float64 {
	out := make([]float64, max(k, 1))
	for i := range out {
		out[i] = 1 / float64(len(out))
	}
	return out
}

// PredictClassProba returns the most likely class and its probability in a
// single pass over the ensemble — cheaper than calling PredictClass and Predict
// separately (each of which walks every tree).
func (m *Model) PredictClassProba(x []float64) (int, float64) {
	if len(x) < m.NumFeature {
		return 0, 1 / float64(max(m.NumClass, 1))
	}
	if m.NumClass == 1 {
		p := sigmoid(m.rawBinary(x))
		if p >= 0.5 {
			return 1, p
		}
		return 0, 1 - p
	}
	prob := make([]float64, m.NumClass)
	softmax(m.rawScores(x), prob)
	best := 0
	for c := 1; c < len(prob); c++ {
		if prob[c] > prob[best] {
			best = c
		}
	}
	return best, prob[best]
}

// PredictBatch returns the probability vector for each row of X.
func (m *Model) PredictBatch(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, x := range X {
		out[i] = m.Predict(x)
	}
	return out
}

// PredictClassBatch returns the predicted class index for each row of X.
func (m *Model) PredictClassBatch(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		out[i] = m.PredictClass(x)
	}
	return out
}

// PredictClass returns the most likely class index (0/1 for Binary).
func (m *Model) PredictClass(x []float64) int {
	if len(x) < m.NumFeature {
		return 0
	}
	if m.NumClass == 1 {
		if m.rawBinary(x) >= 0 { // sigmoid(raw)>=0.5 ⟺ raw>=0
			return 1
		}
		return 0
	}
	raw := m.rawScores(x)
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

// Load reads and validates a model written by Save. A structurally unsound
// model (bad class counts, out-of-range features, non-forward child links) is
// rejected here so Predict can never panic on a corrupt or hostile file.
func Load(r io.Reader) (*Model, error) {
	var m Model
	if err := json.NewDecoder(r).Decode(&m); err != nil {
		return nil, err
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	return &m, nil
}

// validate checks the invariants Predict relies on.
func (m *Model) validate() error {
	if m.Version > modelVersion {
		return fmt.Errorf("grove: model version %d newer than supported %d", m.Version, modelVersion)
	}
	if m.NumClass < 1 {
		return fmt.Errorf("grove: num_class %d < 1", m.NumClass)
	}
	if len(m.Base) != m.NumClass {
		return fmt.Errorf("grove: base has %d scores, want num_class %d", len(m.Base), m.NumClass)
	}
	if m.NumFeature < 0 {
		return fmt.Errorf("grove: num_feature %d < 0", m.NumFeature)
	}
	for r, round := range m.Rounds {
		if len(round) != m.NumClass {
			return fmt.Errorf("grove: round %d has %d trees, want %d", r, len(round), m.NumClass)
		}
		for c := range round {
			if err := validateTree(round[c], m.NumFeature); err != nil {
				return fmt.Errorf("grove: round %d class %d: %w", r, c, err)
			}
		}
	}
	return nil
}

// validateTree confirms a tree is a finite, forward-linked DAG (children are
// appended after their parent during growth, so every child index is > its
// parent) with in-range feature references.
func validateTree(t tree, numFeature int) error {
	if len(t.Nodes) == 0 {
		return errors.New("empty tree")
	}
	for i, n := range t.Nodes {
		if n.Leaf {
			continue
		}
		if n.Feature < 0 || n.Feature >= numFeature {
			return fmt.Errorf("node %d feature %d out of range [0,%d)", i, n.Feature, numFeature)
		}
		if n.Left <= i || n.Left >= len(t.Nodes) || n.Right <= i || n.Right >= len(t.Nodes) {
			return fmt.Errorf("node %d has invalid child links l=%d r=%d (nodes=%d)", i, n.Left, n.Right, len(t.Nodes))
		}
	}
	return nil
}
