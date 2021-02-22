package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

type hexArrayTool struct {
	FirstLength bool
	*command
}

func newHexArrayTool(c *command) *hexArrayTool {
	return &hexArrayTool{command: c}
}

func (h *hexArrayTool) flags(cmd *cobra.Command) {
	flags := cmd.Flags()
	flags.BoolVar(&h.FirstLength, "first-length", false, "first number is array length")
}

func (h *hexArrayTool) RunE(cmd *cobra.Command, args []string) error {
	in, err := os.Open(h.input)
	if err != nil {
		return fmt.Errorf("open input file: %w", err)
	}
	defer in.Close()

	out, err := os.Create(h.output)
	if err != nil {
		return fmt.Errorf("open output file: %w", err)
	}
	defer out.Close()

	switch h.mode {
	case "encode":
		if err := h.encode(in, out); err != nil {
			return fmt.Errorf("encode: %w", err)
		}
	case "decode":
		if err := h.decode(in, out); err != nil {
			return fmt.Errorf("decode: %w", err)
		}
	default:
		panic("unknown mode")
	}

	return nil
}

func (h *hexArrayTool) encode(r io.Reader, w io.Writer) error {
	var (
		num string
		buf = make([]byte, 4)
	)

	for {
		_, err := fmt.Fscanf(r, "%s", &num)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return nil
		}

		res, err := strconv.ParseUint(num, 16, 32)
		if err != nil {
			return err
		}

		binary.BigEndian.PutUint32(buf, uint32(res))
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
}

func (h *hexArrayTool) decode(r io.Reader, w io.Writer) error {
	buf := make([]byte, 4)
	if h.FirstLength {
		_, err := r.Read(buf)
		if err != nil {
			return err
		}
		length := binary.BigEndian.Uint32(buf)
		tail := " "
		if length == 0 {
			tail = "\n"
		}
		if _, err := fmt.Fprintf(w, "%08X"+tail, length); err != nil {
			return err
		}
		if length == 0 {
			return nil
		}
	}

	var length int
	for ; ; length++ {
		n, err := r.Read(buf)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		if n != len(buf) {
			return errors.New("fucked")
		}
		var prefix string
		if length != 0 {
			prefix = " "
		}
		num := binary.BigEndian.Uint32(buf)
		if _, err := fmt.Fprintf(w, prefix+"%08X", num); err != nil {
			return err
		}
	}
}
