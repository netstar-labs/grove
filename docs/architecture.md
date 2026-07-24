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

Continuous features are mapped to at most `MaxBins` (≤255, leaving a `uint8` slot
for the missing bin) ordinal bins via **quantile edges** — for a high-cardinality column, split
points chosen so each bin holds a roughly equal share of the data; a column with
fewer than `MaxBins` distinct values simply gets one bin per value. A constant
column collapses to one bin. Bins
exist only during training: each tree node stores the real edge value as its
threshold, so `Predict` compares raw features and never needs the binner. The
binned matrix is stored **feature-major** (`bt[feature][sample]`) so building a
feature's histogram is a linear, cache-friendly pass.

## Tree growth (`tree.go`)

Each tree is grown depth-first to `MaxDepth`. A node's histogram — per-feature,
per-bin sums of gradients and hessians, stored as one flat array with stride
`maxBins` — is built once; the best split is found by scanning it left→right
tracking the cumulative left sum, maximizing the regularized gain

```
gain = ½ [ GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ) ]
```

subject to `MinChildWeight` (minimum child hessian) and `Gamma` (minimum gain).
If no split clears the bar, the node becomes a leaf with the regularized optimal
weight `−G/(H+λ)`, shrunk by the learning rate. The gain of each split is recorded
on the node — that's what powers `Importance()`.

**Histogram subtraction.** On a split, only the *smaller* child's histogram is
built from its samples; the *larger* child's is derived by subtracting it from
the parent's (`large = parent − small`). Since the smaller child holds ≤ half the
samples, this roughly halves the histogram-building work per level. Node
histograms come from a small **arena** (a free-list of flat buffers) shared
across the whole forest, so subtraction reuses memory instead of allocating per
node. (Subtraction introduces floating-point drift versus a fresh build, which
can flip a near-tie split — accepted, as in the mainstream GBDT engines; fits
stay deterministic.)

## Boosting (`boost.go`)

The ensemble is additive. `raw[i]` holds sample `i`'s running score per class,
seeded with a constant base (label log-odds for binary; per-class log-frequency
for multiclass; the target mean for regression). Each round computes the loss
gradient/hessian at the current `raw`, fits one tree per class to those
gradients, and adds the tree's (shrunk) output back into `raw`. Logistic loss
gives `g = p − y`, `h = p(1−p)`; softmax gives the per-class analogue; regression
(squared error) gives `g = pred − y`, `h = 1`. Optional row `Subsample` (seeded) adds stochastic
regularization. The whole loop is deterministic for a fixed `Seed`.

## Model + inference (`model.go`)

A `Model` is pure data: the base scores and `Rounds[][]tree` (one tree per class
per round; one per round for binary), plus optional feature/class names.
`Predict` re-accumulates `base + Σ trees` and maps through sigmoid/softmax;
`PredictClass` returns the argmax. `Importance()` sums each feature's split gains
across the ensemble. Models serialize to compact JSON (`Save`/`Load`) — inspect
them, embed them, diff them across retrains.

## Parallel histogram building

For a large node (≥2048 samples) on a wide feature set (≥16 features), the
histogram *fill* fans out across `GOMAXPROCS` workers — each owns a disjoint
stripe of features, and therefore a disjoint index range of the flat histogram,
so the workers write in parallel with no lock and no race (`go test -race`). The
result is independent of worker count, so the fit stays deterministic. Narrow
fits and small (deep) nodes stay serial and never pay the goroutine tax. With
histogram subtraction already halving the fill work, the parallel path is a
smaller additional win on top.

## Trade-offs

- **Depth-first, exact histograms with subtraction.** O(n·d·depth) per tree,
  halved per level by parent−sibling subtraction. Fine for in-memory corpora; a
  persistent worker pool (to cut per-node goroutine spawns) is the next lever.
- **`float64` throughout.** Accuracy and simplicity over the memory/speed of
  `float32`.
- **Missing values (NaN)** get their own bin per feature and a learned default
  direction per split (the side that maximized gain during training), so a
  missing feature at predict time routes deterministically. Bins are capped at
  255 to reserve the missing slot within a `uint8`. Clean data is unaffected —
  the missing bin is empty, so splits are identical to having no missing bin.
- **Single-machine, in-memory.** No distribution, no out-of-core. Deliberate: the
  target is inline scoring and modest corpora, not leaderboard scale.
