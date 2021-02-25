package main

import (
	"fmt"
	"io"
	"os"
)

func openTwo(input, output string, cb func(r io.Reader, w io.Writer) error) error {
	var (
		r io.Reader = os.Stdin
		w io.Writer = os.Stdout
	)
	if input != "" {
		in, err := os.Open(input)
		if err != nil {
			return fmt.Errorf("open input file: %w", err)
		}
		defer in.Close()
		r = in
	}

	if output != "" {
		out, err := os.Create(output)
		if err != nil {
			return fmt.Errorf("open output file: %w", err)
		}
		defer out.Close()
		w = out
	}

	return cb(r, w)
}
