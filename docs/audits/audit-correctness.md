# Audit — correctness + optimization (auditor C), fresh A1 pass

Scope: grove.go, tree.go, boost.go, model.go, bins.go, app/grove/{main,helpers}.go,
pkg/serve/serve.go, pkg/mcp/mcp.go, example/*. Builds on the prior audit-and-fix
cycle (audit(correctness), audit(simplify/dedup), audit(docs), audit(v0.1.2)) —
none of that ground re-litigated except where noted. The connectors (pkg/serve,
pkg/mcp) were the highest-value target: newer, network-facing, less scrutinized
than the tree/boost core.

## CONFIRMED and fixed (each independently reproduced by an adversarial skeptic
before being fixed, against a throwaway copy, never the repo source)

### 1. [SECURITY, critical] Params.NumClass/Rounds had no upper bound

`Params.fill()` (grove.go) enforced `NumClass >= 2` for Multiclass but no upper
bound; `Rounds` had no bound in either direction beyond `<= 0` taking the
default (unlike `MaxBins`, which was already clamped to 255). `NumClass` drives
several allocations directly proportional to its value (`baseScores`'s
`cnt`/`base` slices, `prob`, and the per-class `gc`/`hc` slice-of-slices) plus a
`for c := 0; c < k; c++` per-class tree-building loop each round; `Rounds`
drives the outer round loop with no cancellation hook anywhere in `Fit`.

Reachable with no authentication via `pkg/serve`'s `POST /train` and the MCP
`grove_train` tool, from a request under 200 bytes
(`{"params":{"objective":"multiclass","num_class":2000000000},...}`).

**Skeptic-verified, and found worse than initially reported**: measured ~600MB
heap growth at `NumClass = 2,000,000` (far more than the naive "two k-sized
slices" estimate — `gc`/`hc` are `n×k` in aggregate, not `k`), and a
**second, independent DoS vector**: a `NumClass` far too small to trigger OOM
(≈10^8) instead hangs the handling goroutine indefinitely inside the serial
per-class tree-building loop — no crash needed, no memory pressure, just an
unkillable request.

**Fix**: `fill()` now rejects `NumClass > 10000` and `Rounds > 1000000` with a
clean error, before any k-sized structure is allocated or the round loop starts.
Regression test added; **the sabotage-verification run for this fix was itself
a live demonstration of the vulnerability** — with the bound removed, the test
process was killed by the OS before the fix was restored.

### 2. [BUG] MCP `grove_load` tool was unusable by any schema-conformant caller

`case "grove_load": payload, err = bind(call.Arguments, s.core.Load)` — `bind`
unmarshals the JSON-RPC `arguments` object directly into `Req`, inferred here
as `string` (`Load`'s parameter type). But the tool's own registered schema
advertises an *object*, `{"name": "..."}`. Every well-formed call — one that
follows the tool's own contract — failed with `json: cannot unmarshal object
into Go value of type string`. The sibling `grove_save` tool was hand-special-
cased with its own struct specifically to avoid this exact trap; `grove_load`
wasn't.

**Skeptic-verified** by building the real binary and running real JSON-RPC
sessions: trained and saved a real model, confirmed the file existed on disk,
then confirmed `grove_load` still failed on it — a 100% reproducible failure,
not an edge case.

**Fix**: `grove_load` now binds through the same object-shaped `Req` pattern as
`grove_save` (via a wrapping closure); `grove_save` was simplified to use
`bind` too instead of hand-rolling the same shape separately. Regression test
added (`TestMCPLoadRoundTrip`), sabotage-verified against the exact original
error message.

### 3. [BUG] pkg/mcp had no panic recovery anywhere in its dispatch chain

Confirmed via `grep -rn "recover("` across the whole repo: zero matches,
including in the shared `pkg/serve` core both transports use. The HTTP
transport survives a handler panic only because `net/http.(*conn).serve`
recovers per-connection (framework behavior, not app code) — the stdio MCP
transport has no equivalent, and `mcpCmd` runs `Serve` in the main goroutine, so
an unhandled panic there kills the entire process outright.

**Skeptic-verified live**, using the (then still-open) NumClass panic as the
trigger: the identical payload that fails one HTTP request (server logs the
panic via `net/http`'s own recover, stays up, next request succeeds normally)
kills the whole MCP process (exit code 2, stderr stack trace, session over).

**Fix**: `withRecover` wraps `Serve`'s per-request dispatch, turning any panic
into a JSON-RPC `-32603` internal-error response instead of letting it unwind
out of the loop. Regression test (`TestWithRecoverContainsAPanic`) exercises
the containment mechanism directly with a synthetic panic, since the specific
live trigger this session found is now closed by fix #1 — sabotage-verified
(removing the recover reproduces the real panic/crash in the test run).

### 4. [PERF] Double allocation in the hottest recursive path

`tree.go`'s split partition allocated two independently-growing slices
(`left`, `right`), each reserving the *full* parent index-set capacity, even
though together they only ever hold that same total — up to 2x the necessary
backing memory, paid at every non-leaf node of every tree of every round.

**Fix**: one `make([]int, len(idx))`, filled from both ends. Implemented as a
two-pass count-then-fill (not a single back-filling pass) specifically to keep
each side's element order identical to before: `buildHist`'s histogram
accumulation is float summation, not order-independent, and this package's
documented contract ("a seeded fit is reproducible") requires bit-for-bit
determinism. Verified via `TestGoldenPredictions`, unchanged before/after.

## MINOR — fixed

- **`pkg/serve` `hLoad` returned 404 for both a syntactically invalid model
  name (rejected by `path()`'s own validation, never touches disk) and a valid
  name with no file behind it.** Now 400 for the former, 404 only for the
  latter (`errors.Is(err, os.ErrNotExist)`). Regression test added,
  sabotage-verified.
- **`app/grove predict()` ignored two `csv.Writer.Write` return values** — a
  broken output pipe would silently truncate output instead of surfacing an
  error. Fixed.
- **`app/grove eval()` re-derived the predicted-class index via a
  `slices.Index` string search** when it was already in hand from
  `PredictClass`, in the common case where `classes == m.Classes`. Fixed, but
  gated: the fallback branch (a model loaded without stored class names, where
  `classes` is rebuilt from the data's own label order — a *different* index
  space than `predIdx`) keeps the original search, since substituting the
  index there would silently change behavior rather than just remove
  redundant work.

  **Correction, found by the final whole-diff skeptic**: the first version of
  this fix's gate ("common case where `classes == m.Classes`") was itself
  wrong — `classes == m.Classes` does not imply `predIdx` is a valid index
  into it. Nothing enforces `len(m.Classes) == m.NumClass` anywhere in the
  codebase (`pkg/serve.Train` sets `m.Classes = req.Classes` with no length
  check), so a model trained over the network with a short `Classes` list
  loads and predicts fine, but this "optimization" would panic the moment a
  row predicted a class index past the end of the stored names — reproduced
  live (train→save→eval, panic at current HEAD vs. no panic at the pre-pass
  commit, confirming the regression was introduced here). Fixed by falling
  back to the search whenever `predIdx` would be out of range, not just in
  the already-handled "unnamed" branch — mirrors the bounds guard
  `pkg/serve.Predict` already has for the identical reason. Regression test
  added (`TestEvalHandlesShortClassesList`), sabotage-verified against the
  exact panic. See the `fix:` commit following this pass's `cc6a06d`.
- **`pkg/serve`'s `path()` had a `name == ""` clause fully subsumed by
  `name != filepath.Base(name)`** (`filepath.Base("") == "."` always, verified
  directly with a throwaway Go program, not assumed from documentation).
  Removed.
- **`boost.go` computed `baseScores` for Regression only to immediately
  discard it** in favor of the training-target mean. Fixed to skip the
  computation.
- **`boost.go`'s early-stop validation split allocated a second `0..n-1`
  permutation and discarded the first** — the original `trainIdx` was never
  used before being overwritten. Fixed to shuffle `trainIdx` in place (same
  final permutation, since `Shuffle`'s output depends only on `n` and the
  seed, not on which backing array holds the initial content).

## REFUTED (verified, not silently dropped)

**Claim**: `MaxDepth` uncapped upward + degenerate training data → deep
recursion in `grow()` → `fatal error: stack overflow`.

**Refuted by an independent skeptic** with adversarial empirical testing
(added regression cases to a throwaway copy, including deliberately stripping
regularization via extreme `Gamma`/`Lambda`/`MinChildWeight` to force splits at
every node): recursion is bounded independently of `MaxDepth` by (a) `grow`'s
own `len(idx) < 2` / `bestF < 0` halts, and more importantly (b) a hard
architectural ceiling — feature bins are `uint8`-encoded and `MaxBins` is
capped at ≤255, so **any single feature can never support more than ~254
levels** of "isolate one more row" splitting, no matter how large `MaxDepth`
or the training set. The claim's own proposed repro (a single degenerate
feature) cannot escape this ceiling by construction. Deliberately adversarial
attempts topped out at depth 73 of a ~253 theoretical ceiling. Not fixed — no
fix needed.

## Clean checks performed (traced, not assumed)

- `pkg/serve.Server`'s `mu sync.RWMutex` correctly guards the swappable
  `*grove.Model` pointer with no check-then-act race: `current()` RLocks and
  returns a snapshot; `Train`/`Load` build a private, unpublished model before
  the single `Lock()`-protected swap.
- `Model.validate()` rejects a hostile/corrupt model file (out-of-range
  feature indices, non-forward child links, version gate) before `Predict` can
  ever be reached with bad data.
- `tree.go`'s histogram parallelism partitions the flat arena buffer by
  disjoint per-feature offset ranges — confirmed race-free by `-race` and by
  reading the offset-assignment code.
- `path()`'s traversal restriction (`[-_.a-zA-Z0-9]` only, no separators) holds
  even allowing `.` in the character class — `..foo` is not `..`, verified.
- NaN-label validation gap in `boost.go`'s Multiclass path (an out-of-range
  check that NaN can defeat on some architectures) is real at the Go-API
  level but **not reachable** through either wire protocol: both `decode`
  (HTTP) and `bind` (MCP) reject `NaN`/`Infinity` as invalid JSON before the
  core method is ever called (verified directly against `encoding/json`).
  Not fixed this pass — a library-API-misuse concern, not a network-facing one.

## Re-validation gate

`go build`/`vet`/`staticcheck`/`deadcode`/`gofmt -l` clean; `go test -race
./...` green across all four packages after every commit, including
`TestGoldenPredictions` (proves the dedup/simplify changes moved zero bits of
observable output) and all four fuzz targets. Every fix carries a
sabotage-verified regression test (revert → confirm an attributable failure →
restore → confirm green); the NumClass/Rounds and MCP-panic sabotage runs were
themselves live reproductions of the underlying vulnerability, not just test
failures.
