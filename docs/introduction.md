# grove

*A stand of trees, each planted in the shade of the last.*

One decision tree guesses — blunt, loud, usually wrong. Plant a hundred, each
grown to fix what the last one botched, and the wood starts to think. That's
gradient boosting. That's **grove**: not a forest of independent guesses but a
stand of *corrections* — tree after tree, a hundred small mistakes cancelling
into one clean verdict.

**What it actually is.** A from-scratch gradient-boosted decision tree library in
pure Go — no C, no Python, no third-party anything. It bins each feature into a
handful of ranges, walks per-bin gradient histograms for the split that cuts the
loss hardest, and does it again, round after round, shrinking every tree so the
ensemble *converges* instead of lurching. Binary or multiclass. And the model it
hands back is **just data** — a base score and a list of trees you can serialize
to JSON, diff in a pull request, and read with your own eyes.

**Trains in Go, scores in Go — one binary, no runtime.** Every rival worth using
is C++ wearing a Python coat: superb for training, a nightmare to *ship*. grove
has no other half. A model fit offline drops onto a locked-down box and scores a
message the instant it lands — inline, no interpreter, no native library, not one
packet of network. Train where the data lives; predict where the traffic runs;
carry the same static binary either way.

**And it shows its work.** Every split records the gain it bought, so
`Importance()` hands you the real ledger of what the wood leaned on — not a
shrug. A boosted forest is usually a black box; grove keeps the lights on,
because a score you can't explain is a score you can't trust.

**It picks its fights.** grove is small on purpose. It trades the last point of
accuracy and the last microsecond for legibility and reproducible, seeded fits.
Training over a billion rows across a cluster? Go call the C++ giants. A few dozen
features, scored inline, zero dependencies, on whatever host you've got? That's
grove's whole reason to exist — every knob that matters (depth, rounds, learning
rate, regularization), none of the ceremony.

---

*Read next → [executive summary](executive-summary.md) · [architecture](architecture.md) · [user guide](userguide.md)*
