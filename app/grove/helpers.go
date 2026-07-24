package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/netstar-labs/grove"
)

func flagset(name string) *flag.FlagSet { return flag.NewFlagSet(name, flag.ExitOnError) }

// readCSV loads a CSV with a header row, returning the header and data rows.
func readCSV(path string) (header []string, rows [][]string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(recs) == 0 {
		return nil, nil, fmt.Errorf("%s: no header row", path)
	}
	return recs[0], recs[1:], nil
}

// columns resolves the target column index and the feature columns (everything
// not the target and not ignored), preserving header order.
func columns(header []string, target string, ignore map[string]bool) (cols []int, names []string, tCol int, err error) {
	tCol = indexOf(header, target)
	if tCol < 0 {
		return nil, nil, -1, fmt.Errorf("target column %q not in header", target)
	}
	for i, h := range header {
		if i == tCol || ignore[h] {
			continue
		}
		cols = append(cols, i)
		names = append(names, h)
	}
	if len(cols) == 0 {
		return nil, nil, -1, fmt.Errorf("no feature columns after dropping target/ignored")
	}
	return cols, names, tCol, nil
}

// alignFeatures maps a model's feature names onto an input header's columns, so
// predict/eval line up by name regardless of column order.
func alignFeatures(header, featNames []string) ([]int, error) {
	cols := make([]int, len(featNames))
	for j, name := range featNames {
		i := indexOf(header, name)
		if i < 0 {
			return nil, fmt.Errorf("model feature %q missing from input header", name)
		}
		cols[j] = i
	}
	return cols, nil
}

// features extracts a numeric feature vector from a row at the given columns.
func features(row []string, cols []int) ([]float64, error) {
	x := make([]float64, len(cols))
	for j, c := range cols {
		v, err := strconv.ParseFloat(strings.TrimSpace(row[c]), 64)
		if err != nil {
			return nil, fmt.Errorf("column %d (%q) is not numeric", c, row[c])
		}
		x[j] = v
	}
	return x, nil
}

func loadModel(path string) (*grove.Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return grove.Load(f)
}

func className(m *grove.Model, cls int) string {
	if cls >= 0 && cls < len(m.Classes) {
		return m.Classes[cls]
	}
	return strconv.Itoa(cls)
}

func indexOf(ss []string, s string) int {
	for i, v := range ss {
		if v == s {
			return i
		}
	}
	return -1
}

func splitSet(csv string) map[string]bool {
	set := map[string]bool{}
	for _, s := range strings.Split(csv, ",") {
		if s = strings.TrimSpace(s); s != "" {
			set[s] = true
		}
	}
	return set
}

func keys(m map[string]bool) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}

func setFromMap(m map[string]int) map[string]bool {
	s := make(map[string]bool, len(m))
	for k := range m {
		s[k] = true
	}
	return s
}
