package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/pflag"
)

func main() {
	var (
		wantName string
		resName  string
	)
	pflag.StringVar(&wantName, "want", "want", "filename for expected data")
	pflag.StringVar(&resName, "result", "res", "filename for result data")
	pflag.Parse()

	want, err := os.Open(wantName)
	if err != nil {
		panic(err)
	}
	defer want.Close()

	res, err := os.Open(resName)
	if err != nil {
		panic(err)
	}
	defer res.Close()

	wantArr, err := scanNums(want)
	if err != nil {
		panic(err)
	}
	resArr, err := scanNums(res)
	if err != nil {
		panic(err)
	}

	if err := equalMatrix(wantArr, resArr); err != nil {
		println(err.Error())
	}
}

func scanNums(r io.Reader) (res [][]float64, err error) {
	s := bufio.NewScanner(bufio.NewReader(r))
	var lineNum int
	for s.Scan() {
		nums := strings.Split(s.Text(), " ")
		line := make([]float64, 0, len(nums))
		for numID, numS := range nums {
			if numS == "" {
				break
			}
			num, err := strconv.ParseFloat(numS, 64)
			if err != nil {
				return nil, fmt.Errorf("parse float: %d %d: %w", lineNum, numID, err)
			}
			line = append(line, num)
		}
		lineNum++
		res = append(res, line)
	}
	return res, nil
}

func equalMatrix(a, b [][]float64) error {
	if len(a) == 0 || len(b) == 0 {
		return fmt.Errorf("empty matrix: %d and %d", len(a), len(b))
	}
	if len(a[0]) == 0 || len(b[0]) == 0 {
		return fmt.Errorf("empty matrix lines: %d and %d", len(a[0]), len(b[0]))
	}
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		return fmt.Errorf("dimensions mismatch: %dx%d and (%dx%d)", len(a), len(a[0]), len(b), len(b[0]))
	}
	for i, line := range a {
		for j := range line {
			if math.Abs(a[i][j]-b[i][j]) > 1e-8 {
				return fmt.Errorf("numbers mismatch: [%d;%d] %f %f", i, j, a[i][j], b[i][j])
			}
		}
	}
	return nil
}
