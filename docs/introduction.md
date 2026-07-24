# grove

*A stand of trees, each planted in the shade of the last.*

Plant one decision tree and it will give you a blunt, confident, mostly-wrong
answer. Plant a hundred — each one grown to correct the mistakes the ones before
it left behind — and the wood starts to think. That's gradient boosting, and
that's **grove**: an ensemble of small trees where every new tree is a
correction, not a fresh guess, until the whole stand casts one sharp verdict.

**What it actually is.** grove is a from-scratch gradient-boosted decision tree
library in pure Go — no C, no Python, no third-party anything. It bins each
feature into a handful of ranges, walks per-bin gradient histograms to find the
split that most reduces the loss, and repeats, round after round, shrinking each
tree's contribution so the ensemble converges instead of overfitting on the
first swing. It trains binary (logistic) and multiclass (softmax) targets, and
the model it produces is *just data*: a base score and a list of trees you can
serialize to JSON, diff in a pull request, and read with your own eyes.

**The thing nobody else in this stack gives you.** The model trains in Go and
**scores in Go** — the same toolchain, the same static binary, no runtime to
install. A model fit offline on a corpus drops straight onto an Alpine box and
scores a message *inline*, at the moment of capture, without ever reaching for
the network or a Python interpreter. Train where the data is; predict where the
traffic is; ship one binary either way.

**And it won't lie to you about why.** Every split records the gain it bought, so
`Importance()` tells you which features the wood actually leaned on — not a
hand-wave, the real ledger. A boosted forest is often a black box; grove keeps
the lights on, because a score you can't explain is a score you can't trust.

**The honest scope.** grove is deliberately small. It favours legibility and
reproducible, seeded fits over squeezing out the last point of accuracy or the
last microsecond — if you need distributed training over a billion rows, reach
for the C++ giants. For a few dozen features and a corpus that fits in memory,
scored inline with no dependencies, grove is the right size of tool. You keep the
knobs that matter (depth, rounds, learning rate, regularization) and none of the
ceremony.

---

*Read next → [executive summary](executive-summary.md) · [architecture](architecture.md) · [user guide](userguide.md)*
