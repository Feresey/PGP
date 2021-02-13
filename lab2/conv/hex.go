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

type hexTool struct {
	*command
}

func newHexTool(c *command) *hexTool {
	return &hexTool{command: c}
}

func (h *hexTool) RunE(cmd *cobra.Command, args []string) error {
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

func (h *hexTool) encode(r io.Reader, w io.Writer) error {
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
			return err
		}

		res, err := strconv.ParseUint(num, 16, 32)
		if err != nil {
			return err
		}

		binary.LittleEndian.PutUint32(buf, uint32(res))
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
}

func (h *hexTool) decode(r io.Reader, w io.Writer) error {
	header := make([]byte, 8)
	if _, err := r.Read(header); err != nil {
		return fmt.Errorf("read dimensions: %w", err)
	}

	width := binary.BigEndian.Uint32(header[:4])
	height := binary.BigEndian.Uint32(header[4:])

	if _, err := fmt.Fprintf(w, "%08x %08x\n", width, height); err != nil {
		return err
	}

	var currWidth uint32

	buf := make([]byte, 4)
	for {
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
		currWidth++
		var tail string
		if currWidth == width {
			currWidth = 0
			tail = "\n"
		} else {
			tail = " "
		}
		num := binary.LittleEndian.Uint32(buf)
		if _, err := fmt.Fprintf(w, "%08x"+tail, num); err != nil {
			return err
		}
	}
}
