// Model lifecycle: fit with named features and classes, save to bytes, reload,
// and batch-predict with human-readable labels — the metadata + save/load path
// the CLI uses internally, shown from the library. Also exercises early stopping.
//
//	GOWORK=off go run ./example/roundtrip
package main

import (
	"bytes"
	"fmt"
	"math/rand"

	"github.com/netstar-labs/grove"
)

func main() {
	featureNames := []string{"url_count", "has_password_input", "wallet_present", "noise"}
	classes := []string{"bulk", "phish", "scam"}

	// synthetic labelled data: a password field ⇒ phish, a wallet ⇒ scam, else bulk.
	rng := rand.New(rand.NewSource(7))
	const n = 3000
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		urls := float64(rng.Intn(6))
		pw := float64(rng.Intn(2))
		wallet := float64(rng.Intn(2))
		cls := 0.0 // bulk
		if pw == 1 {
			cls = 1 // phish
		}
		if wallet == 1 {
			cls = 2 // scam
		}
		X[i] = []float64{urls, pw, wallet, rng.Float64()}
		y[i] = cls
	}

	m, err := grove.Fit(X, y, grove.Params{
		Objective: grove.Multiclass, NumClass: len(classes),
		Rounds: 300, MaxDepth: 4, LearningRate: 0.15,
		EarlyStop: 15, // let the fit find its own length
	})
	if err != nil {
		panic(err)
	}
	m.FeatureNames = featureNames
	m.Classes = classes

	// save → reload: the on-disk model must score identically.
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		panic(err)
	}
	loaded, err := grove.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		panic(err)
	}
	fmt.Printf("fit %d trees (early-stopped), saved %d bytes, reloaded model v%d\n",
		loaded.TreeCount(), buf.Len(), loaded.Version)

	sample := [][]float64{
		{2, 0, 0, 0.5}, // expect bulk
		{1, 1, 0, 0.5}, // expect phish
		{0, 0, 1, 0.5}, // expect scam
	}
	for i, cls := range loaded.PredictClassBatch(sample) {
		fmt.Printf("  %v -> %s\n", sample[i], classes[cls])
	}

	fmt.Println("feature importance:")
	imp := loaded.Importance()
	for i, name := range featureNames {
		fmt.Printf("  %-20s %.1f\n", name, imp[i])
	}
}
