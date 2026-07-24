# grove — architecture

grove is a flat library at the repo root (`package grove`) plus a thin CLI under
`app/grove`. Five files, one concern each. This doc walks the pipeline from raw
features to a scored prediction.

## Data flow

```
X [][]float64 ─▶ fitBinner ─▶ binner.edges ─┐
                                             ├─▶ transpose ─▶ bt [][]uint8 (feature-major)
y []float64 ────────────────────────────────┘
                              │
        baseScores ─▶ raw[n][K] (running per-class score, init = base)
                              │
        for round in Rounds:
          for class c in K:
            gradients(raw, y, c) ─▶ g[i], h[i]         (logistic or softmax deriv.)
            buildTree(bt, edges, g, h, idx) ─▶ tree    (histogram split finding)
            raw[i][c] += tree.predict(X[i])            (add the correction)
                              │
        Model{Base, Rounds:[][]tree} ─▶ Predict / PredictClass / Importance
```

## Binning (`bins.go`)

Continuous features are mapped to at most `MaxBins` (≤256, so a `uint8` holds a
bin) ordinal bins via **quantile edges** — for a high-cardinality column, split
points chosen so each bin holds a roughly equal share of the data; a column with
fewer than `MaxBins` distinct values simply gets one bin per value. A constant
column collapses to one bin. Bins
exist only during training: each tree node stores the real edge value as its
threshold, so `Predict` compares raw features and never needs the binner. The
binned matrix is stored **feature-major** (`bt[feature][sample]`) so building a
feature's histogram is a linear, cache-friendly pass.

## Tree growth (`tree.go`)

Each tree is grown depth-first to `MaxDepth`. At a node, for every feature we
accumulate a per-bin histogram of gradients and hessians over the node's samples,
then scan bins left→right tracking the cumulative left sum; the best split
maximizes the regularized gain

```
gain = ½ [ GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ) ]
```

subject to `MinChildWeight` (minimum child hessian) and `Gamma` (minimum gain).
If no split clears the bar, the node becomes a leaf with the regularized optimal
weight `−G/(H+λ)`, shrunk by the learning rate. The gain of each chosen split is
recorded on the node — that's what powers `Importance()`.

## Boosting (`boost.go`)

The ensemble is additive. `raw[i]` holds sample `i`'s running score per class,
seeded with a constant base (label log-odds for binary; per-class log-frequency
for multiclass). Each round computes the loss gradient/hessian at the current
`raw`, fits one tree per class to those gradients, and adds the tree's (shrunk)
output back into `raw`. Logistic loss gives `g = p − y`, `h = p(1−p)`; softmax
gives the per-class analogue. Optional row `Subsample` (seeded) adds stochastic
regularization. The whole loop is deterministic for a fixed `Seed`.

## Model + inference (`model.go`)

A `Model` is pure data: the base scores and `Rounds[][]tree` (one tree per class
per round; one per round for binary), plus optional feature/class names.
`Predict` re-accumulates `base + Σ trees` and maps through sigmoid/softmax;
`PredictClass` returns the argmax. `Importance()` sums each feature's split gains
across the ensemble. Models serialize to compact JSON (`Save`/`Load`) — inspect
them, embed them, diff them across retrains.

## Parallel split finding

For a large node (≥2048 samples) on a wide feature set (≥16 features), split
finding fans out across `GOMAXPROCS` workers — each scans a stripe of features
into its own histogram scratch, and the per-feature bests are reduced **in
feature order**, so the parallel result is bit-identical to the serial one and
the fit stays deterministic. Narrow fits and small (deep) nodes stay serial, so
they never pay the goroutine tax. On 40 features / 10 cores this cuts fit time
~20%; the shared training data is read-only and result slots are disjoint, so
the pass is race-free (`go test -race`).

## Trade-offs

- **Depth-first, exact histograms, no histogram subtraction (yet).** Simple and
  correct; O(n·d·depth) per tree. Fine for in-memory corpora; the next
  optimization is parent−sibling histogram subtraction (roughly halves the
  histogram-building work), and a persistent worker pool to cut per-node
  goroutine spawns.
- **`float64` throughout.** Accuracy and simplicity over the memory/speed of
  `float32`.
- **No missing-value handling.** Features are assumed present (the inbox matrix
  has no gaps); NaN is out of scope.
- **Single-machine, in-memory.** No distribution, no out-of-core. Deliberate: the
  target is inline scoring and modest corpora, not leaderboard scale.
