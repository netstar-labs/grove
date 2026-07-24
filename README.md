# grove

A small, dependency-free **gradient-boosted decision tree** library in pure Go —
train binary/multiclass/regression models offline, score them inline anywhere Go runs.

```
  features (n×d) ─▶ quantile bins ─▶ ┌──────────── boosting rounds ────────────┐
                                     │  gradients/hessians ─▶ histogram split   │
                                     │  ▲                         │            │
                                     │  └── residual ◀── grow tree ─┘  × Rounds │
                                     └──────────────────────────────────────────┘
                                                     │
                              Model {base, trees[][]} ─▶ Predict(x) ─▶ prob / class
                                                     └─▶ Importance() ─▶ feature gains
```

The library is the core; `app/grove` is a thin CLI over any CSV feature matrix
(numeric columns + a categorical or numeric target). Training and inference share
one toolchain and zero third-party deps — a model fit offline scores inline on any
host, no Python, no native library.

## Documentation

- **Start here** — [docs/introduction.md](docs/introduction.md) (what it is and
  what the name means) · [docs/executive-summary.md](docs/executive-summary.md)
  (what and why).
- **Deep dive** — [docs/architecture.md](docs/architecture.md) (binning, split
  finding, the boosting loop, the model format, trade-offs).
- **Operations** — [docs/userguide.md](docs/userguide.md) (library API, CLI
  flags, CSV/model formats).
- **Examples** — [example/README.md](example/README.md) (runnable programs).
- **How-to** — [docs/howto.md](docs/howto.md) (step-by-step: library, CLI, and the HTTP/unix/MCP connectors).

## Quick start

```go
m, _ := grove.Fit(X, y, grove.Params{Objective: grove.Binary, Rounds: 120, MaxDepth: 4})
p := m.Predict(x)          // []float64 probabilities
c := m.PredictClass(x)     // argmax class
imp := m.Importance()      // per-feature total split gain
```

```sh
grove train   -in data.csv -target type -out model.json
grove predict -model model.json -in data.csv
grove eval    -model model.json -in data.csv -target type
```

## Layout

| File | Purpose |
|------|---------|
| [`grove.go`](grove.go) | package doc, `Params`, defaults, numeric helpers (sigmoid/softmax) |
| [`bins.go`](bins.go) | quantile feature binning (the histogram substrate) |
| [`tree.go`](tree.go) | regression tree: node/tree types, histogram split finding, growth |
| [`boost.go`](boost.go) | the boosting loop (`Fit`), base scores, logistic/softmax/regression gradients, early stopping |
| [`model.go`](model.go) | `Model`: `Predict`/`PredictClass`/`PredictValue`, batch, `Importance`, versioned JSON save/load |
| [`app/grove/`](app/grove) | CLI: `train`/`predict`/`eval`/`serve`/`mcp` |
| [`pkg/serve/`](pkg/serve) | HTTP + unix-socket connector: train/predict/save/load/model |
| [`pkg/mcp/`](pkg/mcp) | MCP stdio connector (same core) for AI agents |
| [`example/`](example) | runnable demos (library embed, validation harness, model roundtrip) |
| [`build/grove`](build/grove) | cross-compile + package the CLI |

## Notes

- Go module `github.com/netstar-labs/grove` (Go 1.26), **standard library only**.
- Deterministic: a fit with a fixed `Seed` reproduces exactly.
- Build standalone with `GOWORK=off go build ./...`.

## License

GPL-3.0 — see [LICENSE](LICENSE).
