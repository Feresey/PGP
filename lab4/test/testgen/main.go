package main

import (
	"bufio"
	"io"
	"os"
	"strconv"

	"github.com/spf13/pflag"
)

func main() {
	var (
		n, m       int
		output     string
		transponse bool
	)
	pflag.IntVarP(&n, "n-rows", "n", 1, "")
	pflag.IntVarP(&m, "m-rows", "m", 1, "")
	pflag.StringVarP(&output, "output", "o", "-", "output file")
	pflag.BoolVar(&transponse, "transponse", false, "")
	pflag.Parse()

	var out io.Writer = os.Stdout
	if output != "-" {
		outFile, err := os.Create(output)
		if err != nil {
			panic(err)
		}
		defer outFile.Close()
		out = outFile
	}

	o := bufio.NewWriterSize(out, 1<<20)
	defer o.Flush()

	if transponse {
		n, m = m, n
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if j != 0 {
				_ = o.WriteByte(' ')
			}
			var num int
			if transponse {
				num = j*m + i
			} else {
				num = i*m + j
			}
			_, _ = o.WriteString(strconv.Itoa(num))
		}
		_ = o.WriteByte('\n')
	}
}
