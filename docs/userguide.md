# grove — user guide

## Library

```go
import "github.com/netstar-labs/grove"

// X is row-major n×d float64; y is 0/1 (Binary) or class indices (Multiclass).
m, err := grove.Fit(X, y, grove.Params{
    Objective:    grove.Binary, // or grove.Multiclass with NumClass
    Rounds:       120,
    MaxDepth:     4,
    LearningRate: 0.2,
})

prob  := m.Predict(x)       // []float64: {P(1)} for Binary, distribution for Multiclass
class := m.PredictClass(x)  // int argmax
imp   := m.Importance()     // []float64 per-feature total split gain

m.Save(w)                   // JSON
m2, _ := grove.Load(r)
```

### Params

| Field | Default | Meaning |
|---|---|---|
| `Objective` | `Binary` | `Binary` (logistic) or `Multiclass` (softmax) |
| `NumClass` | — | number of classes (Multiclass only) |
| `Rounds` | 100 | boosting rounds (Binary: 1 tree/round; Multiclass: `NumClass`/round) |
| `LearningRate` | 0.1 | shrinkage per tree |
| `MaxDepth` | 6 | max tree depth |
| `MaxBins` | 256 | histogram bins per feature (2..256) |
| `Lambda` | 1 | L2 regularization on leaf weights |
| `Gamma` | 0 | minimum gain to split |
| `MinChildWeight` | 1 | minimum summed hessian in a child |
| `Subsample` | 1 | row fraction sampled per tree |
| `Seed` | 1 | RNG seed (subsampling); a fit is reproducible for a fixed seed |

Unset (zero) fields are filled with the defaults above, so `grove.Fit(X, y,
grove.Params{Objective: grove.Binary})` is valid.

## CLI (`app/grove`)

The CLI operates on CSV feature matrices with a header row — the shape
`netstar-labs/inbox`'s `tools/train` emits. Non-feature columns are dropped with
`-ignore`; every remaining column must be numeric.

```sh
# train — auto-selects binary (2 classes) or multiclass (>2)
grove train -in data.csv -target type -ignore source,rule_score -out model.json \
    -rounds 200 -depth 5 -lr 0.1 -subsample 0.8 -seed 1

# predict — emits source,predicted,probability as CSV
grove predict -model model.json -in data.csv

# eval — accuracy + actual→predicted confusion against a labelled column
grove eval -model model.json -in data.csv -target type

grove version
```

### train flags

| Flag | Default | Meaning |
|---|---|---|
| `-in` | — | input CSV with a header (required) |
| `-target` | — | label column name (required) |
| `-ignore` | `source,rule_score` | comma list of columns to drop before training |
| `-out` | `model.json` | model output path |
| `-rounds` / `-depth` / `-lr` | 100 / 6 / 0.1 | boosting rounds / max depth / learning rate |
| `-subsample` / `-seed` | 1.0 / 1 | row subsample fraction / RNG seed |

`predict` and `eval` align input columns to the model's stored feature names by
name, so column order in the input need not match training.

## Model format

A model is JSON: `objective`, `num_class`, `num_feature`, `base` (per-class
scores), `learning_rate`, `rounds` (a list of rounds, each a list of trees; a
tree is a flat `nodes` array), and optional `feature_names` / `classes`. It is
inspectable and diffable — a retrain's changes show up in a `git diff`.

## Building

```sh
GOWORK=off go build ./...          # standalone (grove has no workspace siblings)
GOWORK=off go test ./...           # unit tests + validation
build/grove                        # cross-compile + package the CLI (build/install/)
build/grove user@host              # also scp + install on a host
```
