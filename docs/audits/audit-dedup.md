# Audit — duplication / dedup (auditor B), fresh A1 pass

Scope: all non-test `.go` files. The connectors (`pkg/serve`, `pkg/mcp`,
`app/grove`) were the highest-value target — newer than the core, less
scrutinized by the prior dedup pass.

## Applied

### 1. Cross-package Binary/Multiclass distribution logic (highest value)

Three independent re-derivations of "which class does this distribution
pick" / "what's this distribution's probability for class idx", all encoding
the same Binary-vs-Multiclass convention:

- `model.go` `PredictClassProba` — canonical, but works from raw scores, not
  an already-computed `dist`.
- `pkg/serve/serve.go`'s `Predict` — an inline block deriving the class from
  an already-computed `dist` (deliberately inlined in a prior commit to avoid
  a second ensemble pass — the fix below had to preserve that).
- `app/grove/main.go`'s `probOfClass` — the inverse operation (probability of
  a given class from an already-computed `dist`), with its own copy of the
  same Binary special-case.

**Fix**: `Model.ClassOf(dist)` and `Model.ProbOf(dist, idx)` added, both
operating on an already-computed `dist` (preserving the single-ensemble-pass
property each call site exists to protect). `pkg/serve` and `app/grove` both
call the new methods; `probOfClass` deleted.

**Risk note acted on**: the two duplicate sites' argmax loops differed in
loop-start detail (`k := 1` vs `argmax`'s own loop) — confirmed they agree on
ties before merging (both use strict `>` starting from index 1, first-max
wins), and `TestGoldenPredictions` passing after the change is direct
evidence they do.

### 2. File-based model load/save boilerplate (4 sites)

The "open/create, defer Close, encode/decode, propagate error" idiom, repeated
identically at `app/grove/helpers.go` (`loadModel`), `app/grove/main.go`
(`train`'s save), and twice in `pkg/serve/serve.go` (`Load`, `persist`).

**Fix**: `Model.SaveFile(path)` / `grove.LoadFile(path)` added next to
`Save`/`Load`, a natural extension of the existing `io.Writer`/`io.Reader`
API for callers that don't already hold one. All four sites now call through
it.

### 3. Model swap-under-lock (2 sites)

`s.mu.Lock(); s.model = m; s.mu.Unlock()` appeared verbatim in `Train` and
`Load`.

**Fix**: `Server.setModel(m)`. Small LOC win, but specifically worth doing
because it's exactly the kind of lock-pattern duplication a future third call
site (a hot-swap/reload endpoint) would otherwise copy a third time.

### 4. `clamp`/`clampInt`/`clampProb` — see `audit-simplify.md`

The library-internal merge (`clamp`/`clampInt`) was applied. `app/grove`'s
`clampProb` was deliberately **not** merged into it — doing so would require
exporting the generic `clamp` from the library for a one-line CLI-local
helper, a bigger public-API-surface decision than this duplication warrants.

## Considered and NOT recommended

- **`post[Req,Resp]` (HTTP) vs `bind[Req,Resp]` (MCP)** — structurally similar
  generics but encode genuinely different transport concerns: `post` wraps a
  full `http.HandlerFunc` with status-code mapping and a size-capped body
  reader; `bind` is a bare unmarshal-then-call for the JSON-RPC layer to wrap
  itself. Merging would save ~5 lines at the cost of coupling `pkg/mcp` to
  HTTP-shaped decode semantics it doesn't need.
- **CLI subcommand required-flag checks** (`train`/`predict`/`eval`/`serve`/
  `mcp`) — each checks a different flag combination with a purpose-specific
  message; a generic "require these flags" helper would save ~1 line per site
  at the cost of a new indirection and genericized error text.
- **`example/{embed,roundtrip,validate}` synthetic-data loops** — each
  encodes a genuinely different labeling function and pedagogical point
  (XOR demo, multiclass demo, boundary-recovery validation gate). Standalone
  demo mains by design (each runs with no shared deps); a shared generator
  would serve exactly 3 call sites and obscure the point of each example.
- **`hSave`/`hLoad`/`hInfo` HTTP handlers** — already minimal (3-8 lines
  each) and not uniform (`hSave` has a status-code branch the others don't
  need). A shared wrapper saves ~2 lines on one of three handlers.
- **Wire types** (`TrainRequest`/`Response`, `PredictRequest`/`Response`,
  `ModelInfo`) are correctly **not** duplicated — `pkg/mcp` reuses `pkg/serve`'s
  types directly via generic inference, exactly one definition of each shape
  shared by both connectors. Checked, not assumed.

## Re-validation

See `audit-correctness.md`'s gate section — shared across all four dimensions
of this pass.
