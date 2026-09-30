# grove — pre-flight audit findings

A fresh A1 torture-chamber pass (four report-only auditors + least-code, each
run independently, then an adversarial-verification pass that tried to
**refute** every candidate before any of it was believed) on top of the prior
audit-and-fix cycle recorded in this repo's git history
(`audit(correctness)`, `audit(simplify/dedup)`, `audit(docs)`,
`audit(v0.1.2)`). **4 CONFIRMED (1 critical security, 2 bugs, 1 perf) · 1
REFUTED · several dedup/simplify/doc fixes applied.** Full per-dimension
detail in `audit-simplify.md` / `audit-dedup.md` / `audit-correctness.md` /
`audit-docs.md`.

## CONFIRMED and fixed (each reproduced by an independent skeptic before
being fixed, against a throwaway copy of the repo — never the source)

- **[security, critical] `Params.NumClass`/`Rounds` had no upper bound** —
  reachable with no auth via `POST /train` and the MCP `grove_train` tool
  from a ~150-byte payload; drives O(NumClass) and O(Rounds×NumClass)
  allocations/loops with no cancellation hook. Skeptic found it worse than
  initially reported: a CPU-hang vector (no crash, no memory pressure)
  independent of the OOM vector. Fixed with sane ceilings in `Params.fill()`.
  See `audit-correctness.md` #1.
- **[bug] MCP `grove_load` tool was unusable by any schema-conformant
  caller** — a generic-inference mismatch between the tool's own advertised
  schema and what `bind` actually unmarshals into. Reproduced by building
  the real binary and running real JSON-RPC sessions, including a full
  train→save→load round trip that still failed to load. See
  `audit-correctness.md` #2.
- **[bug] pkg/mcp had no panic recovery anywhere** — the same bug that fails
  one HTTP request kills the entire MCP session outright. Reproduced live
  with the (then still-open) NumClass panic as the trigger. See
  `audit-correctness.md` #3.
- **[perf] Double allocation in the hottest recursive path** (tree.go's split
  partition) — one allocation instead of two, order-preserving (verified via
  `TestGoldenPredictions`, since a seeded fit must stay bit-for-bit
  reproducible). See `audit-correctness.md` #4.

## REFUTED — verified, not silently dropped

- **`MaxDepth` uncapped → stack-overflow DoS via degenerate data.** An
  independent skeptic's adversarial empirical testing (including
  deliberately stripping regularization) found a hard architectural ceiling
  — `uint8` histogram bin storage + `MaxBins≤255` caps any single feature at
  ~254 split levels regardless of `MaxDepth` — that the claim's own proposed
  repro cannot escape. Topped out at depth 73 of a ~253 ceiling. Not fixed;
  no fix needed. See `audit-correctness.md`'s Refuted section.

## MINOR — fixed alongside the above

`pkg/serve`'s `hLoad` conflated a bad name (400) with a missing file (404);
`app/grove predict()` ignored two CSV-writer errors; `eval()` re-derived an
index already in hand (gated to the branch where that's actually safe);
`path()` had one dead boolean clause. See `audit-correctness.md`.

## Dedup / simplify / docs — applied, no behavior change

- `clamp`/`clampInt` merged into one generic (found independently by 3
  passes — least-code, simpler-pathways, dedup).
- `Model.ClassOf`/`ProbOf` replace three independent re-derivations of the
  Binary/Multiclass distribution convention across `pkg/serve` and
  `app/grove`.
- `Model.SaveFile`/`LoadFile` replace four hand-rolled file-I/O sites.
- `Server.setModel` replaces two copies of the same lock+swap+unlock.
- `boost.go`: a discarded `baseScores` computation and a redundant
  index-permutation allocation, both removed.
- Doc fixes: an overclaimed "every snippet runs," an inverted description of
  where the missing-value bin sits relative to `MaxBins`, an undocumented
  numerical floor in `gradHess`.

Full lists of what was considered and deliberately **not** changed (with
reasons) are in each per-dimension doc — load-bearing complexity, subtly
different near-duplicates, and API-surface decisions that weren't worth
making for a small duplication.

## Re-validation gate

`go build`/`vet`/`staticcheck`/`deadcode`/`gofmt -l` clean; `go test -race
./...` green across all four packages after every commit, including
`TestGoldenPredictions` and all four fuzz targets. Every fix carries a
sabotage-verified regression test (revert → confirm an attributable failure
→ restore → confirm green) — two of those sabotage runs were themselves live
reproductions of the underlying vulnerability (an OS-killed runaway process
for NumClass/Rounds; a real crash for the MCP panic-recovery removal).

## Final whole-diff skeptic — caught a regression this pass introduced

Before this pass was called done, an independent skeptic reviewed the entire
diff fresh (not re-trusting the earlier per-finding skeptics) and found that
the `eval()` fix in `cc6a06d` (above) had itself introduced a real panic: its
gating condition assumed `classes == m.Classes` implies `predIdx` is always a
valid index into it, which nothing in the codebase actually guarantees.
Reproduced live (A/B against the pre-pass commit), fixed, sabotage-verified —
see `audit-correctness.md`'s correction note and the `fix:` commit below.
Everything else the skeptic independently re-verified (the security fixes,
the golden-prediction determinism, the dedup, the fuzz targets) held up.

This is the reason the final skeptic step exists: a pass's own confidence in
its fixes isn't the same as those fixes being correct, and a "verify with a
skeptic" step that only re-confirms what earlier skeptics already believed
would have missed this.

Commits: `510b6c2` (security), `cc6a06d` (dedup/simplify/correctness),
`5eeb0b5` (docs), `ae42822` (audit docs), `98571db` (fuzz targets), `b6b5d14`
(L1/L2 summary), `27b254a` (fix: the eval() regression above).
