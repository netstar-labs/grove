# Audit — doc / comment vs code drift (auditor D), fresh A1 pass

Scope: README.md, docs/*.md, example/README.md, every package/function doc
comment. Builds on a prior doc-drift fix cycle. Bottom line: **no misleading
or outdated drift found** — everything below is cosmetic. The connectors'
docs (the newest material) were, if anything, the most carefully cross-checked
part of the doc set (exact route tables, exact JSON shapes, exact tool names).

## Fixed

1. **`docs/howto.md`'s "every snippet runs" overclaimed.** The library
   snippets are illustrative fragments (reference undefined `X`, `y`, `x`),
   unlike the CLI/curl snippets and the three `example/` programs, which
   genuinely do run as shown. Reworded; pointed at `example/` for complete
   runnable versions.
2. **The missing-value bin was described as carved out of `MaxBins`'s own
   count** ("a slot is reserved... within `MaxBins`") across
   `docs/architecture.md`, `docs/userguide.md`, and `grove.go`'s `Params`
   comments. Verified against the actual code (`tree.go`: `nb[f] = len(e) +
   2`, "regular bins (len+1) + one missing bin") — the missing bin is
   **additional** to the up-to-255 regular bins; the 255 cap exists only so
   the total (256) still fits a `uint8` index. Corrected everywhere the
   phrasing appeared.
3. **`boost.go`'s `gradHess` floors the hessian at `1e-6`** (keeps it off
   zero at the p≈0/1 extremes) — `docs/architecture.md` described the
   formula (`h = p(1−p)`) without the floor. Documented.

## Noted, not a fix

- **The MCP `grove_train` tool's advertised schema for `params` is an
  opaque `{"type":"object"}`** rather than enumerating `grove.Params`'s
  fields. Not a doc/code mismatch — the code's own comment already says
  "Schemas are intentionally light," and `docs/userguide.md` already
  discloses that `params` mirrors the `Params` struct by field name. Worth
  knowing an MCP client relying strictly on the schema (not the prose docs)
  won't discover the field names from it, but nothing here contradicts
  anything else.

## Clean checks performed (verified, not skipped)

- **Dependencies**: `go.mod` has no `require` block; `go list -m all` lists
  only the module itself. "Zero dependencies / pure Go" claims are literally
  true.
- **License**: LICENSE is Apache-2.0, `Copyright 2026 NetSTAR Global, Inc.`;
  no lingering GPL text or private-repo references anywhere in tracked
  files.
- **Version numbers**: no doc references a stale version anywhere. The MCP
  `serverInfo.version: "0.1"` is a hardcoded protocol string, independent of
  the git-tag-stamped CLI version — no doc claims otherwise.
- **HTTP routes** (`POST /train`, `/predict`, `/save`, `/load`, `GET
  /model`) exactly match `docs/userguide.md`/`docs/howto.md`, including
  request/response JSON shapes, field-by-field.
- **`params` JSON casing**: `grove.Params` has no json tags (wire keys are
  literal Go field names, PascalCase) — the docs' curl examples get this
  right, a real and easy-to-get-wrong asymmetry against the snake_case outer
  envelope.
- **MCP tool set and CLI subcommands/flags**: match the docs exactly,
  including required-field lists and the `-ignore` default.
- **Training/scoring algorithm** (binning, histogram-subtraction, the exact
  gain formula, `Gamma`/`MinChildWeight` gating, missing-value routing,
  parallel-histogram thresholds): traced line-by-line against
  `docs/architecture.md` — exact matches, including specific numeric
  thresholds (`parThreshold=2048`, `minParFeatures=16`).
- **Determinism claim**: traced the parallel histogram-fill path (workers
  own disjoint, statically-computed feature ranges) — "independent of
  worker count" verified true, not just plausible.
- **`Params` defaults/`fill()` table**: every row checked against the
  zero-value-fallback logic, including the `Lambda` "0→default,
  negative→none" special case.
- **Build tooling** (`build/grove`, `build/test`): matches script contents
  exactly, including the git-describe version stamping.
- Ran the actual build, vet, and full test suite — all green, so no
  code-doesn't-compile-style drift either.

## Re-validation

See `audit-correctness.md`'s gate section — shared across all four
dimensions of this pass.
