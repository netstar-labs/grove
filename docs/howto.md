# grove — how to use it, step by step

A tour of every path: the library (binary, multiclass, regression, importance,
save/load, batch, missing values, early stopping), the CLI over CSV, and the
HTTP / unix / MCP connectors. Every snippet runs; build standalone with
`GOWORK=off`.

## 1. Library — binary classification

```go
import "github.com/netstar-labs/grove"

// X: row-major n×d; y: 0/1.
m, err := grove.Fit(X, y, grove.Params{
    Objective: grove.Binary, Rounds: 200, MaxDepth: 4, LearningRate: 0.1,
})
p     := m.Predict(x)[0]    // P(class = 1)
class := m.PredictClass(x)  // 0 or 1
imp   := m.Importance()     // per-feature total split gain
```

## 2. Multiclass

```go
// y: class indices in [0, NumClass).
m, _ := grove.Fit(X, y, grove.Params{Objective: grove.Multiclass, NumClass: 3, Rounds: 200, MaxDepth: 4})
dist  := m.Predict(x)      // length-3 probability distribution
class := m.PredictClass(x) // argmax
```

## 3. Regression

```go
// y: continuous.
m, _ := grove.Fit(X, y, grove.Params{Objective: grove.Regression, Rounds: 300, MaxDepth: 4, EarlyStop: 20})
v := m.PredictValue(x)     // the predicted value
```

## 4. Early stopping

```go
// Hold out 20% for validation; stop after 15 rounds with no held-out
// improvement, then keep the best round.
m, _ := grove.Fit(X, y, grove.Params{
    Objective: grove.Binary, Rounds: 1000, MaxDepth: 4,
    EarlyStop: 15, ValFraction: 0.2,
})
// m.TreeCount() is well below 1000 — it found its own length.
```

## 5. Missing values

A `NaN` feature is fine — it gets its own bin and a learned default direction:

```go
import "math"
X[i][3] = math.NaN()       // "missing" — no need to impute
m, _ := grove.Fit(X, y, grove.Params{Objective: grove.Binary, Rounds: 100})
// A NaN feature at predict time follows the split's learned direction.
```

## 6. Save, load, batch score

```go
var buf bytes.Buffer
m.Save(&buf)                       // indented, diffable JSON (versioned, validated on load)
m2, _ := grove.Load(&buf)          // rejects corrupt/hostile models

classes := m2.PredictClassBatch(X) // []int over many rows
probs   := m2.PredictBatch(X)      // [][]float64
```

Set human labels so predictions read back by name:

```go
m.FeatureNames = []string{"url_count", "has_form", "wallet"}
m.Classes = []string{"bulk", "phish", "scam"}
```

Runnable: [`example/embed`](../example/embed), [`example/roundtrip`](../example/roundtrip),
[`example/validate`](../example/validate).

## 7. CLI over CSV

Any CSV with a header; non-feature columns are dropped with `-ignore`.

```sh
# data.csv:  id,f0,f1,f2,label
grove train   -in data.csv -target label -ignore id -out model.json \
    -rounds 200 -depth 5 -earlystop 12
grove predict -model model.json -in data.csv          # source,predicted,probability
grove eval    -model model.json -in data.csv -target label
#   accuracy, log loss, per-class precision/recall/F1, macro-F1, confusion
```

## 8. Connectors — HTTP / unix / MCP

One model server; three transports over the same core.

### HTTP (or unix socket)

```sh
grove serve -http :8080 -dir models         # add -unix /run/grove.sock to also serve there
```

```sh
# receive raw data → train → save
curl -s localhost:8080/train -d '{
  "params": {"Objective":"multiclass","NumClass":3,"Rounds":120,"MaxDepth":4},
  "features": [[0.1,0.5],[0.9,0.2], ...],
  "labels":   [0,2, ...],
  "classes":  ["low","mid","high"],
  "save":     "demo"
}'
# → {"objective":"multiclass","num_class":3,"trees":360,"classes":[...],"importance":[...],"saved":"demo"}

# reload into any server (survives a restart) and score
curl -s -XPOST 'localhost:8080/load?name=demo'
curl -s localhost:8080/predict -d '{"features":[[0.9,0.2]]}'
# → {"classes":[2],"labels":["high"],"probabilities":[[...]]}

curl -s localhost:8080/model     # metadata: objective, classes, trees, importance
```

Endpoints: `POST /train`, `POST /predict`, `POST /save?name=`, `POST /load?name=`,
`GET /model`. Model names must be plain base names (no path separators).

### MCP (for AI agents)

```sh
grove mcp -dir models
```

Newline-delimited JSON-RPC 2.0 on stdin/stdout, tools: `grove_train`,
`grove_predict`, `grove_save`, `grove_load`, `grove_model_info`. Arguments mirror
the HTTP request bodies; results come back as a JSON text content block.

## 9. Scale — how big before this stops fitting in memory

grove is single-machine, in-memory `float64`. Footprint ≈ `9·n·d` bytes (the
feature matrix dominates); fit time ≈ `rounds·K·n·d·depth` (halved by histogram
subtraction).

| Tier | Size | Footprint | Notes |
|---|---|---|---|
| Comfortable | ≤ ~1M rows × ~50–100 feat (n·d ≲ 10⁸) | a few GB | seconds–a minute |
| Large (float32 would help) | ~1M–10M rows × 100+ feat (n·d ~ 10⁸–10⁹) | 5–40 GB | minutes; strains RAM |
| Beyond grove | > ~10M rows / tens of GB | > RAM | use out-of-core / a distributed engine |

See [architecture.md](architecture.md#scale) for the memory model.
