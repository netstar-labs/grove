package grove

import (
	"slices"
	"sort"
)

// binner maps each feature's continuous values onto a small set of ordinal bins
// via quantile edges. Split search then runs over per-bin gradient histograms
// instead of every raw value — the histogram trick that keeps boosting fast.
//
// Bins are chosen at fit time and never leave training: tree nodes store the
// real edge value as their threshold, so Predict compares raw features directly
// and needs no binner.
type binner struct {
	edges [][]float64 // per feature: ascending upper thresholds, len = nBins-1
}

// fitBinner computes quantile bin edges per feature over X (row-major, n×d).
func fitBinner(X [][]float64, maxBins int) *binner {
	d := 0
	if len(X) > 0 {
		d = len(X[0])
	}
	b := &binner{edges: make([][]float64, d)}
	col := make([]float64, len(X))
	for f := 0; f < d; f++ {
		for i := range X {
			col[i] = X[i][f]
		}
		b.edges[f] = quantileEdges(col, maxBins)
	}
	return b
}

// quantileEdges returns up to maxBins-1 ascending split points for one column.
// A constant column yields no edges (a single bin).
func quantileEdges(col []float64, maxBins int) []float64 {
	vals := append([]float64(nil), col...)
	sort.Float64s(vals)
	uniq := slices.Compact(vals) // sorted, so this is a full de-dup in place
	if len(uniq) <= 1 {
		return nil
	}
	// few distinct values: split midway between each adjacent pair
	if len(uniq) <= maxBins {
		e := make([]float64, len(uniq)-1)
		for i := 0; i+1 < len(uniq); i++ {
			e[i] = midpoint(uniq[i], uniq[i+1])
		}
		return e
	}
	// many distinct values: quantile edges, de-duplicated and ascending
	nEdges := maxBins - 1
	e := make([]float64, 0, nEdges)
	for i := 1; i <= nEdges; i++ {
		pos := float64(i) / float64(maxBins) * float64(len(uniq)-1)
		lo := int(pos)
		v := uniq[lo]
		if lo+1 < len(uniq) {
			v = midpoint(uniq[lo], uniq[lo+1])
		}
		if len(e) == 0 || v > e[len(e)-1] {
			e = append(e, v)
		}
	}
	return e
}

func midpoint(a, b float64) float64 { return a + (b-a)/2 }

// binValue returns the bin index of v under feature f: the first bin whose upper
// edge is >= v, so "bin <= i" is exactly equivalent to "v <= edges[i]".
func (b *binner) binValue(f int, v float64) uint8 {
	e := b.edges[f]
	i := sort.Search(len(e), func(i int) bool { return v <= e[i] })
	return uint8(i)
}

// transpose bins X into feature-major columns for cache-friendly histogramming.
func (b *binner) transpose(X [][]float64) [][]uint8 {
	d := len(b.edges)
	n := len(X)
	bt := make([][]uint8, d)
	for f := 0; f < d; f++ {
		col := make([]uint8, n)
		for i := 0; i < n; i++ {
			col[i] = b.binValue(f, X[i][f])
		}
		bt[f] = col
	}
	return bt
}
