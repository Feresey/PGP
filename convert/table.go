package main

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"text/template"

	"github.com/spf13/cobra"
)

type tableTool struct {
	output string
	input  string
	skip   int
	sort   string
	delim  string
	desc   bool
	num    bool
}

func newTableCommand() *cobra.Command {
	var t tableTool
	cmd := &cobra.Command{
		Use:  "table",
		RunE: t.RunE,
	}
	t.flags(cmd)
	return cmd
}

func (t *tableTool) flags(cmd *cobra.Command) {
	flags := cmd.Flags()

	flags.StringVarP(&t.input, "input", "i", "", "Input file")
	flags.StringVarP(&t.output, "output", "o", "", "Output file")
	flags.IntVar(&t.skip, "skip", 0, "Number of lines to skip from beginning of the file")
	flags.StringVarP(&t.sort, "sort", "s", "", "Column to sort by")
	flags.StringVarP(&t.delim, "delimiter", "d", "=", "Delimiter character")
	flags.BoolVar(&t.desc, "desc", false, "Sort order")
	flags.BoolVar(&t.num, "num", false, "Sort numbers")
}

func (t *tableTool) RunE(cmd *cobra.Command, args []string) error {
	tb := Table{
		columnsIndex: make(map[string]int),
	}

	return openTwo(t.input, t.output, func(r io.Reader, w io.Writer) error {
		if err := tb.Parse(r, t.skip, t.delim); err != nil {
			return fmt.Errorf("scan: %w", err)
		}

		if t.sort != "" {
			tb.Sort(t.sort, !t.desc, t.num)
		}

		tb.Lines = transponse(tb.Lines)
		if err := tb.Format(w); err != nil {
			return fmt.Errorf("write: %w", err)
		}

		return nil
	})
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

func (tb *Table) Parse(r io.Reader, skip int, delim string) error {
	scan := bufio.NewScanner(r)
	for scan.Scan() {
		if skip > 0 {
			skip--
			continue
		}
		vals := strings.Split(scan.Text(), delim)
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

func (tb *Table) Sort(column string, ascending, asNum bool) {
	idx, ok := tb.columnsIndex[column]
	if !ok {
		println("no such column:", column)
		return
	}

	swapIndexes := swapIndex(tb.Lines[idx], ascending, asNum)

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
	asNum     bool
}

func (a sortIndex) Len() int { return len(a.lines) }
func (a sortIndex) Swap(i, j int) {
	a.lines[i], a.lines[j] = a.lines[j], a.lines[i]
	a.indexes[i], a.indexes[j] = a.indexes[j], a.indexes[i]
}

func (a sortIndex) Less(i, j int) bool {
	var res bool
	if a.asNum {
		x, _ := strconv.ParseFloat(a.lines[i], 64)
		y, _ := strconv.ParseFloat(a.lines[j], 64)
		res = x < y
	} else {
		res = a.lines[i] < a.lines[j]
	}
	if a.ascending {
		return res
	} else {
		return !res
	}
}

func swapIndex(ss []string, ascending bool, asNum bool) []int {
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
		asNum:     asNum,
	}

	sort.Stable(tosort)

	return tosort.indexes
}
