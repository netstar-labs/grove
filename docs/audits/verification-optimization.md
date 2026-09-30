# grove — L1/L2 verification & hardening summary

Scope note up front: this is not the full L1/L2 treatise (no per-symbol
coverage ledger, no CPU/memory profiling pass, no formal optimization
priority matrix). The A1 pass immediately preceding this (see
`audit-findings.md`) already did most of L1's mandate as a side effect —
doc-vs-code verification, dead-code mapping, and hardening on the
highest-risk surface (the connectors) — so this covers what's left: running
the existing suite for real, and closing the one concrete coverage gap the
audit pass's own findings pointed at.

## Operational validation (run for real, not assumed)

- `go build`/`vet`/`staticcheck`/`deadcode`/`gofmt -l`: clean.
- `go test -race ./... -count=1`: green across all four packages.
- Coverage: root package 87.6%, `pkg/serve` 79.5%, `pkg/mcp` 76.6%,
  `app/grove` 5.0% (the CLI's `main`/flag-dispatch layer is exercised only
  by its two internal-helper fuzz targets, not integration-tested end to
  end — noted as a gap below, not fixed).
- Benchmarks: `BenchmarkFitWide` sanity-checked stable across 3 runs after
  the `tree.go` allocation fix (133ms/op, ~19MB/op, ~9.3k allocs/op) — no
  regression, no re-baselining done beyond this spot check.
- Fuzzing: all 4 pre-existing targets pass their seed corpora; the 3 new
  targets below were run for real time budgets, not just seeded and left.

## Coverage gap found and closed

**pkg/serve and pkg/mcp — the two network-facing, attacker-reachable
packages, where every bug this session found lived — had zero fuzz
targets**, unlike the core library (which fuzzes `Fit` and the model file
format thoroughly). Closed with three targets (`test(L1/L2)` commit):

| Target | Surface | Execs (this run) | Crashes |
|---|---|---|---|
| `FuzzServe` (pkg/mcp) | raw bytes → JSON-RPC decode → dispatch → every tool arm | 78,634 | 0 |
| `FuzzTrainBody` (pkg/serve) | raw bytes as the `POST /train` body | 1,364,678 | 0 |
| `FuzzLoadName` (pkg/serve) | arbitrary strings through `path()`, asserting the resolved path never escapes the model directory | 2,070,930 | 0 |

These are a genuine fuzzed confirmation that this pass's security fixes
hold under adversarial input generation, not hand-picked cases — not just a
coverage-percentage exercise.

## Hardening checklist (L1 §3.6), scoped to what the audit pass didn't
already cover

- **Input boundaries**: `POST /train`/`/predict` bodies and the MCP tool
  arguments are now fuzzed (above); `NumClass`/`Rounds` are bounded
  (`grove.go`); model names are fuzzed against directory escape.
- **Concurrency**: `pkg/serve.Server`'s model-pointer swap already verified
  race-free (`audit-correctness.md`'s clean checks) and re-confirmed under
  `-race` on every commit this pass.
- **Resource limits**: `NumClass`/`Rounds` ceilings close the two DoS
  vectors found; the HTTP body cap (256 MiB) predates this pass and was
  independently confirmed still in place.
- **Error handling**: the two silently-ignored `csv.Writer.Write` returns
  were the only unchecked-error finding; fixed.
- **Not done**: a systematic goroutine-lifecycle audit of `app/grove
  serveCmd`'s two listener goroutines (HTTP + unix socket) — both are
  fire-and-forget `go func(){ ... }()` with no supervised shutdown path.
  Noted as a real gap, not fixed this pass — closing it means deciding what
  "shut down cleanly" should mean for a CLI-launched long-running server
  (signal handling, drain timeout), which is a design question, not a bug
  fix.

## Untestable-without-X

- **`app/grove`'s CLI dispatch layer (`main.go`)** has real line coverage
  only from two narrow fuzz targets (`FuzzFeatures`, `FuzzSplitSet`) against
  internal helpers — the actual `train`/`predict`/`eval`/`serve`/`mcp`
  subcommand flows have no direct unit tests, only the two example programs
  exercising the library API directly (not the CLI binary). Needs: a
  `cmd`-level test harness that invokes `run(args)` (or shells out to the
  built binary) against `t.TempDir()` fixtures — a real but larger build-out
  than this pass's scope, deferred.
- **`app/grove serveCmd`'s listener goroutines** are only reachable
  end-to-end via `pkg/serve`'s own `Handler()` tests, not through the CLI
  entry point itself (no test drives `serve(args)` and hits a real listening
  socket). Needs: a `t.Cleanup`-scoped helper that starts the real subcommand
  against an ephemeral port/socket and confirms clean shutdown on context
  cancellation or a signal — currently there's no cancellation seam to test
  against (ties back to the goroutine-lifecycle gap above).

## What this deliberately did not do

- No CPU/memory profiling pass (`pprof`) — the one benchmark spot-check
  above was to confirm the `tree.go` allocation fix didn't regress, not a
  systematic hot-path search.
- No formal per-symbol coverage ledger (L2 Phase 0) — the coverage numbers
  above are package-level, not symbol-by-symbol.
- No optimization priority matrix (L1 Phase 5) — no new optimization
  candidates were identified beyond what the A1 correctness pass already
  found and fixed (the tree.go double-allocation).

These are real, larger pieces of work than this pass's mandate ("fix
anything discovered, verify with a skeptic") called for — flagged here so a
future pass has an honest starting point rather than a false "L1/L2 done"
signal.
