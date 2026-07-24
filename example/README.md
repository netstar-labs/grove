# grove examples

Runnable programs, one per way to use the library. Build standalone with
`GOWORK=off` (grove has no workspace siblings).

| Example | What it shows | Run |
|---|---|---|
| [`embed/`](embed) | Use the library directly: fit a model in-process, score rows, read feature importance. | `GOWORK=off go run ./example/embed` |
| [`validate/`](validate) | Validation harness: train grove to recover a **known** decision function and confirm near-100% held-out accuracy. Exits non-zero on regression, so it doubles as a gate. | `GOWORK=off go run ./example/validate` |
| [`roundtrip/`](roundtrip) | Model lifecycle: fit with named features/classes + early stopping, save → reload, batch-predict with labels, and read feature importance. | `GOWORK=off go run ./example/roundtrip` |

For the CLI over CSV feature matrices, see [../docs/userguide.md](../docs/userguide.md):

```sh
grove train   -in data.csv -target type -out model.json
grove predict -model model.json -in data.csv
grove eval    -model model.json -in data.csv -target type
```
