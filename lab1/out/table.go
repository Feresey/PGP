package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"sort"
	"strings"
	"text/template"
)

func main() {
	var (
		output string
		input  string
		skip   int
		sort   string
		desc   bool
	)
	flag.StringVar(&input, "i", "", "Input file")
	flag.StringVar(&output, "o", "", "Output file")
	flag.IntVar(&skip, "skip", 0, "Number of lines to skip from beginning of the file")
	flag.StringVar(&sort, "sort", "", "Column to sort by")
	flag.BoolVar(&desc, "desc", false, "Sort order")
	flag.Parse()

	tb := Table{
		columnsIndex: make(map[string]int),
	}

	var in io.Reader = os.Stdin
	if input != "" {
		ifile, err := os.Open(input)
		if err != nil {
			log.Fatalf("open input: %v", err)
		}
		defer ifile.Close()
		in = ifile
	}

	if err := tb.Parse(in, skip); err != nil {
		log.Fatalf("scan: %v", err)
	}

	var out io.Writer = os.Stdout
	if output != "" {
		file, err := os.Create(output)
		if err != nil {
			log.Fatalf("create output: %v", err)
		}
		defer file.Close()
		out = file
	}

	if sort != "" {
		tb.Sort(sort, !desc)
	}

	tb.Lines = transponse(tb.Lines)
	if err := tb.Format(out); err != nil {
		log.Fatalf("write: %v", err)
	}
}

type Table struct {
	columnsIndex map[string]int

	Columns []string
	Lines   [][]string
}

func (tb *Table) AddToColumn(key, value string) {
	idx, ok := tb.columnsIndex[key]
	if !ok {
		idx = len(tb.Columns)
		tb.Columns = append(tb.Columns, key)
		tb.Lines = append(tb.Lines, nil)
		tb.columnsIndex[key] = idx
	}
	tb.Lines[idx] = append(tb.Lines[idx], value)
}

func (tb *Table) Parse(r io.Reader, skip int) error {
	scan := bufio.NewScanner(r)
	for scan.Scan() {
		if skip > 0 {
			skip--
			continue
		}
		vals := strings.Split(scan.Text(), "=")
		if len(vals) != 2 {
			return fmt.Errorf("incorrect format: %q", scan.Text())
		}
		tb.AddToColumn(strings.TrimSpace(vals[0]), strings.TrimSpace(vals[1]))
	}

	return scan.Err()
}

const outTmpl = `
\hline
{{$c := counter}}
{{- range .Columns}}
{{- if call $c}} & {{end}}
{{- printf "%s " .}}
{{- end}} \\
\hline
{{range .Lines}}
{{- $c := counter}}
{{range .}}
{{- if call $c}} & {{end}}
{{- printf "%s " .}}
{{- end}} \\
{{- end}}
\hline
`

func (tb *Table) Format(w io.Writer) error {
	t := template.Must(
		template.New("table").
			Funcs(template.FuncMap{
				"transponse": transponse,
				"counter":    counter,
			}).
			Parse(outTmpl))

	return t.Execute(w, tb)
}

func transponse(matrix [][]string) [][]string {
	n := len(matrix)
	if n == 0 {
		return nil
	}
	m := len(matrix[0])

	res := make([][]string, m)

	for i := 0; i < m; i++ {
		res[i] = make([]string, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			res[j][i] = matrix[i][j]
		}
	}
	return res
}

func counter() func() int {
	i := -1
	return func() int {
		i++
		return i
	}
}

func (tb *Table) Sort(column string, ascending bool) {
	idx, ok := tb.columnsIndex[column]
	if !ok {
		println("no such column:", column)
		return
	}

	swapIndexes := swapIndex(tb.Lines[idx], ascending)

	for lineNum := range tb.Lines {
		line := tb.Lines[lineNum]
		res := make([]string, len(line))
		for idx, swap := range swapIndexes {
			res[idx] = line[swap]
		}
		tb.Lines[lineNum] = res
	}
}

type sortIndex struct {
	lines     []string
	indexes   []int
	ascending bool
}

func (a sortIndex) Len() int { return len(a.lines) }
func (a sortIndex) Swap(i, j int) {
	a.lines[i], a.lines[j] = a.lines[j], a.lines[i]
	a.indexes[i], a.indexes[j] = a.indexes[j], a.indexes[i]
}
func (a sortIndex) Less(i, j int) bool {
	if a.ascending {
		return a.lines[i] < a.lines[j]
	}
	return a.lines[i] > a.lines[j]
}

func swapIndex(ss []string, ascending bool) []int {
	scp := make([]string, len(ss))
	copy(scp, ss)

	indexes := make([]int, len(ss))
	for idx := range ss {
		indexes[idx] = idx
	}

	tosort := sortIndex{
		lines:     scp,
		indexes:   indexes,
		ascending: ascending,
	}

	sort.Stable(tosort)

	return tosort.indexes
}
