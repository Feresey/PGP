package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"

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

	bufout := bufio.NewWriterSize(out, 1<<20)
	defer bufout.Flush()

	switch h.mode {
	case "encode":
		if err := h.encode(in, bufout); err != nil {
			return fmt.Errorf("encode: %w", err)
		}
	case "decode":
		if err := h.decode(in, bufout); err != nil {
			return fmt.Errorf("decode: %w", err)
		}
	default:
		panic("unknown mode")
	}

	return nil
}

func (h *hexTool) encode(r io.Reader, w io.Writer) error {
	var (
		num uint32
		buf = make([]byte, 4)
	)

	for {
		_, err := fmt.Fscanf(r, "%x", &num)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return fmt.Errorf("scan number: %w", err)
		}

		binary.BigEndian.PutUint32(buf, num)
		// binary.BigEndian.PutUint32(buf, binary.LittleEndian.Uint32(buf))
		if _, err := w.Write(buf); err != nil {
			return fmt.Errorf("write res: %w", err)
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

	if _, err := fmt.Fprintf(w, "%08X %08X\n", width, height); err != nil {
		return err
	}

	var currWidth uint32
	width = binary.LittleEndian.Uint32(header[:4])
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
		num := binary.BigEndian.Uint32(buf)
		if _, err := fmt.Fprintf(w, "%08X"+tail, num); err != nil {
			return err
		}
	}
}
