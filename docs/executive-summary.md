# grove — executive summary

**What it is.** grove is a self-contained gradient-boosted decision tree (GBDT)
library written in pure Go, with a small CLI and HTTP/unix/MCP connectors. It
learns from a table of numeric features — binary or multiclass classification, or
regression — and produces a compact, inspectable model.

**Why it exists.** The best GBDT engines (XGBoost, LightGBM) are C++ with Python
front-ends. They're excellent for offline training but awkward to *deploy*:
scoring a single record inline means shipping a runtime, a native library, or a
model-format bridge. grove trades a slice of top-end accuracy and scale for one
property that matters when the model has to run in production: **one language,
one static binary, zero dependencies, train and score in the same place.** A
model fit offline drops onto any host — including a locked-down, no-outbound box
— and scores at the moment data arrives, with no Python and no network.

**Where it fits.** grove is the model stage of a detection or scoring pipeline: a
feature extractor produces a CSV of numeric features per record, grove turns that
into a classifier, and the trained model scores new records inline — beside, or in
place of, a hand-written rule score. The CSV/CLI path is classification; regression
is available through the library and the connectors.

**What you get.**
- Binary (logistic), multiclass (softmax), and regression (squared-error) objectives.
- Early stopping, missing-value handling, and HTTP/unix/MCP connectors.
- Histogram-based training that's fast enough for corpora that fit in memory.
- A JSON model you can serialize, embed, diff, and read.
- **Interpretability by default** — per-feature gain importance, so every model
  can answer "what did you actually use?"
- Reproducibility — a seeded fit is bit-for-bit repeatable.

**What it is not.** Not a distributed trainer, not a GPU engine, not a kitchen
sink of objectives and callbacks. For billions of rows or leaderboard-chasing
accuracy, use the C++ giants. For a few dozen features scored inline with no
dependencies, grove is the right-sized tool.
