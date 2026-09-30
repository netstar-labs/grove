# Audit — simpler pathways (auditor A), fresh A1 pass

Scope: all non-test `.go` files. This dimension also surfaced the `grove_load`
MCP bug (tracked in `audit-correctness.md` #2, since it's a functional defect,
not a style finding) — found while tracing why `bind` exists and whether every
use site actually fits its assumption.

## Applied

- **`clamp`/`clampInt` were the same function per type** (`grove.go`,
  `boost.go`) — merged into one generic `clamp[T int | float64]`. Found
  independently by three passes (least-code, this auditor, the dedup
  auditor) — cross-validated, highest confidence.
- **`boost.go`: `baseScores` computed then discarded for Regression** — see
  `audit-correctness.md`.
- **`boost.go`: training-index permutation allocated twice** in the
  early-stop path — see `audit-correctness.md`.
- **`pkg/serve`'s `path()`: one dead boolean clause** (`name == ""`, fully
  subsumed by `name != filepath.Base(name)`) — verified with a throwaway Go
  program (`filepath.Base("") == "."`), not assumed. See `audit-correctness.md`.

## Considered and NOT simplifiable — load-bearing, don't re-litigate

- **`pkg/serve`'s `path()`: the `name == "."` / `name == ".."` clauses
  themselves.** Technically redundant with the current single call site
  (which always appends `.json` before use, so a bare `.`/`..` name would
  resolve to a harmless filename, not real traversal) — but this is
  deliberate defense-in-depth on a security-sensitive validation routine.
  Removing it would silently depend on every future caller appending a
  suffix. Kept.
- **`pkg/serve.Predict`'s manual class-derivation logic** looked like it
  should just call `Model.PredictClassProba` — but that method computes its
  own raw scores independently and doesn't return the full distribution,
  which `Predict`'s response needs. Calling both would walk the ensemble
  twice per row. (Resolved differently in the dedup pass — see
  `Model.ClassOf`/`ProbOf` in `audit-dedup.md` — which shares the *derivation
  logic* without re-introducing a second ensemble pass.)
- **`tree.go`'s arena pooling, per-feature offset packing, worker fan-out in
  `buildHist`, and histogram-subtraction in `grow`.** The densest code in the
  repo, and every piece is a specific, doc-commented, previously-reviewed
  optimization (arena reuse across the whole forest; per-feature stride so
  one high-cardinality feature can't inflate the others; build-smaller-
  derive-larger histogram subtraction; parallel fan-out gated by measured
  thresholds). Not incidental complexity. Not a finding.
- **`model.go`'s repeated `if m.NumClass == 1 { ... } else { ... }` branches**
  across `Predict`/`PredictClassProba`/`PredictClass`/`PredictValue`. The
  `NumClass == 1` branches route through the allocation-free `rawBinary` fast
  path (no `[]float64` score slice, no softmax call) — the documented reason
  two code paths exist per method. Collapsing them would either lose the
  fast path or reintroduce the duplication one level down.

## Re-validation

See `audit-correctness.md`'s gate section — shared across all four dimensions
of this pass.
