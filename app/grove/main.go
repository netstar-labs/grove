// Command grove trains and applies gradient-boosted-tree models over CSV feature
// matrices — the shape tools/train emits from the inbox corpus, but any numeric
// CSV with a categorical target works. Three subcommands:
//
//	grove train   -in data.csv -target type [-ignore source,rule_score] -out model.json
//	grove predict -model model.json -in data.csv
//	grove eval    -model model.json -in data.csv -target type
//
// train auto-selects binary (2 classes) or multiclass (>2). Non-feature columns
// are dropped with -ignore; the remaining numeric columns are the features, and
// their names are stored in the model so predict/eval align columns by name.
package main

import (
	"encoding/csv"
	"fmt"
	"maps"
	"os"
	"slices"
	"sort"
	"strconv"

	"github.com/netstar-labs/grove"
)

// stamped by build/grove via -ldflags -X.
var (
	version = "dev"
	build   = "none"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "train":
		err = train(os.Args[2:])
	case "predict":
		err = predict(os.Args[2:])
	case "eval":
		err = eval(os.Args[2:])
	case "version", "-version", "--version":
		fmt.Printf("grove %s (%s)\n", version, build)
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "grove:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: grove <train|predict|eval> [flags]  (see -h on each)")
	os.Exit(2)
}

// ---- train -----------------------------------------------------------------

func train(args []string) error {
	fs := flagset("train")
	in := fs.String("in", "", "input CSV with header (required)")
	target := fs.String("target", "", "name of the label column (required)")
	ignore := fs.String("ignore", "source,rule_score", "comma list of columns to drop")
	out := fs.String("out", "model.json", "output model path")
	rounds := fs.Int("rounds", 100, "boosting rounds")
	depth := fs.Int("depth", 6, "max tree depth")
	lr := fs.Float64("lr", 0.1, "learning rate")
	subsample := fs.Float64("subsample", 1.0, "row subsample fraction")
	seed := fs.Int64("seed", 1, "RNG seed")
	fs.Parse(args)
	if *in == "" || *target == "" {
		return fmt.Errorf("train needs -in and -target")
	}

	header, rows, err := readCSV(*in)
	if err != nil {
		return err
	}
	featCols, featNames, tCol, err := columns(header, *target, splitSet(*ignore))
	if err != nil {
		return err
	}

	// encode labels (sorted for determinism)
	classSet := map[string]bool{}
	for _, r := range rows {
		classSet[r[tCol]] = true
	}
	classes := slices.Sorted(maps.Keys(classSet))
	classIdx := map[string]int{}
	for i, c := range classes {
		classIdx[c] = i
	}
	if len(classes) < 2 {
		return fmt.Errorf("target %q has %d classes, need >= 2", *target, len(classes))
	}

	X := make([][]float64, len(rows))
	y := make([]float64, len(rows))
	for i, r := range rows {
		x, err := features(r, featCols)
		if err != nil {
			return fmt.Errorf("row %d: %w", i+2, err)
		}
		X[i] = x
		y[i] = float64(classIdx[r[tCol]])
	}

	p := grove.Params{
		Rounds: *rounds, MaxDepth: *depth, LearningRate: *lr,
		Subsample: *subsample, Seed: *seed,
	}
	if len(classes) == 2 {
		// sorted distinct classes map to {0,1}, so y is already 0/1
		p.Objective = grove.Binary
	} else {
		p.Objective, p.NumClass = grove.Multiclass, len(classes)
	}

	m, err := grove.Fit(X, y, p)
	if err != nil {
		return err
	}
	m.FeatureNames = featNames
	m.Classes = classes

	f, err := os.Create(*out)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := m.Save(f); err != nil {
		return err
	}
	fmt.Printf("trained %s model: %d rows, %d features, %d classes %v, %d trees -> %s\n",
		p.Objective, len(X), len(featNames), len(classes), classes, m.TreeCount(), *out)
	printImportance(m)
	return nil
}

func printImportance(m *grove.Model) {
	imp := m.Importance()
	type fi struct {
		name string
		gain float64
	}
	fis := make([]fi, len(imp))
	for i, g := range imp {
		name := fmt.Sprintf("f%d", i)
		if i < len(m.FeatureNames) {
			name = m.FeatureNames[i]
		}
		fis[i] = fi{name, g}
	}
	sort.Slice(fis, func(i, j int) bool { return fis[i].gain > fis[j].gain })
	fmt.Println("top features by gain:")
	for i := 0; i < len(fis) && i < 10; i++ {
		if fis[i].gain == 0 {
			break
		}
		fmt.Printf("  %-28s %.1f\n", fis[i].name, fis[i].gain)
	}
}

// ---- predict ---------------------------------------------------------------

func predict(args []string) error {
	fs := flagset("predict")
	modelPath := fs.String("model", "model.json", "model path")
	in := fs.String("in", "", "input CSV with header (required)")
	fs.Parse(args)
	if *in == "" {
		return fmt.Errorf("predict needs -in")
	}
	m, header, rows, cols, err := loadModelAndRows(*modelPath, *in)
	if err != nil {
		return err
	}
	srcCol := slices.Index(header, "source")

	w := csv.NewWriter(os.Stdout)
	defer w.Flush()
	w.Write([]string{"source", "predicted", "probability"})
	for _, r := range rows {
		x, err := features(r, cols)
		if err != nil {
			return err
		}
		cls, p := m.PredictClassProba(x) // one ensemble pass for class + probability
		src := ""
		if srcCol >= 0 {
			src = r[srcCol]
		}
		w.Write([]string{src, className(m, cls), strconv.FormatFloat(p, 'f', 4, 64)})
	}
	return nil
}

// ---- eval ------------------------------------------------------------------

func eval(args []string) error {
	fs := flagset("eval")
	modelPath := fs.String("model", "model.json", "model path")
	in := fs.String("in", "", "input CSV with header (required)")
	target := fs.String("target", "", "label column to score against (required)")
	fs.Parse(args)
	if *in == "" || *target == "" {
		return fmt.Errorf("eval needs -in and -target")
	}
	m, header, rows, cols, err := loadModelAndRows(*modelPath, *in)
	if err != nil {
		return err
	}
	if len(rows) == 0 {
		return fmt.Errorf("no data rows in %s", *in)
	}
	tCol := slices.Index(header, *target)
	if tCol < 0 {
		return fmt.Errorf("target column %q not found", *target)
	}

	correct := 0
	confusion := map[string]int{}
	for _, r := range rows {
		x, err := features(r, cols)
		if err != nil {
			return err
		}
		pred := className(m, m.PredictClass(x))
		actual := r[tCol]
		if pred == actual {
			correct++
		}
		confusion[actual+"→"+pred]++
	}
	fmt.Printf("accuracy: %.4f (%d/%d)\n", float64(correct)/float64(len(rows)), correct, len(rows))
	fmt.Println("actual→predicted:")
	for _, k := range slices.Sorted(maps.Keys(confusion)) {
		fmt.Printf("  %-28s %d\n", k, confusion[k])
	}
	return nil
}
