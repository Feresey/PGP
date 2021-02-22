package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
)

type pixelsTool struct {
	*command
}

func newPixelsTool(c *command) *pixelsTool {
	return &pixelsTool{
		command: c,
	}
}

func (p *pixelsTool) writePoints(r io.Reader, w io.Writer) error {
	buf := make([]byte, 4)
	r = bufio.NewReaderSize(r, 1<<20)
	for {
		_, err := r.Read(buf)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return fmt.Errorf("read rgba: %w", err)
		}
		_, err = fmt.Fprintf(w, "%d %d %d\n", buf[0], buf[1], buf[2])
		if err != nil {
			return fmt.Errorf("write points: %w", err)
		}
	}
}

func (p *pixelsTool) decode(r io.Reader, w io.Writer) error {
	buf := make([]byte, 8)
	_, err := r.Read(buf)
	if err != nil {
		return fmt.Errorf("read dismensions: %w", err)
	}

	return p.writePoints(r, w)
}

func (p *pixelsTool) RunE(cmd *cobra.Command, args []string) error {
	in, err := os.Open(p.input)
	if err != nil {
		panic(err)
	}
	defer in.Close()

	out, err := os.Create(p.output)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	return p.decode(in, out)
}
