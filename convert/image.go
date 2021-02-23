package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

type Image struct {
	w, h int

	buf [][]color.RGBA
}

func (i *Image) ColorModel() color.Model { return color.RGBAModel }
func (i *Image) Bounds() image.Rectangle {
	return image.Rectangle{Max: image.Point{X: i.w, Y: i.h}}
}
func (i *Image) At(x, y int) color.Color { return i.buf[y][x] }

func NewImage(r io.Reader, w, h uint32) (image.Image, error) {
	res := &Image{
		w:   int(w),
		h:   int(h),
		buf: make([][]color.RGBA, h),
	}

	buf := make([]byte, 4)
	r = bufio.NewReaderSize(r, 1<<20)
	for y := 0; y < res.h; y++ {
		res.buf[y] = make([]color.RGBA, res.w)
		for x := 0; x < res.w; x++ {
			_, err := io.ReadFull(r, buf)
			pixel := color.RGBA{
				R: buf[0],
				G: buf[1],
				B: buf[2],
				A: buf[3],
			}
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				return nil, fmt.Errorf("read rgba: %w", err)
			}
			res.buf[y][x] = pixel
		}
	}

	return res, nil
}

type imageTool struct {
	*command
}

func newImageTool(c *command) *imageTool {
	return &imageTool{command: c}
}

func (i *imageTool) RunE(cmd *cobra.Command, args []string) error {
	in, err := os.Open(i.input)
	if err != nil {
		return fmt.Errorf("open input file: %w", err)
	}
	defer in.Close()

	out, err := os.Create(i.output)
	if err != nil {
		return fmt.Errorf("open output file: %w", err)
	}
	defer out.Close()

	bufin := bufio.NewReaderSize(in, 1<<20)
	bufout := bufio.NewWriterSize(out, 1<<20)
	defer bufout.Flush()

	switch i.mode {
	case "encode":
		if err := i.encode(bufin, bufout); err != nil {
			return fmt.Errorf("encode: %w", err)
		}
	case "decode":
		if err := i.decode(bufin, bufout, filepath.Ext(i.output)); err != nil {
			return fmt.Errorf("decode: %w", err)
		}
	default:
		panic("unknown mode")
	}

	return nil
}

func (i *imageTool) encode(r io.Reader, w io.Writer) error {
	img, name, err := image.Decode(r)
	println(name)
	if err != nil {
		return fmt.Errorf("decode image: %w", err)
	}
	bounds := img.Bounds()
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint32(buf[:4], uint32(bounds.Max.X-bounds.Min.X))
	binary.LittleEndian.PutUint32(buf[4:], uint32(bounds.Max.Y-bounds.Min.Y))
	_, _ = w.Write(buf)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			_, err := w.Write([]byte{byte(r), byte(g), byte(b), byte(a)})
			if err != nil {
				return fmt.Errorf("write binary: %w", err)
			}
		}
	}

	return nil
}

func (i *imageTool) decode(r io.Reader, w io.Writer, ext string) error {
	buf := make([]byte, 8)
	_, err := io.ReadFull(r, buf)
	if err != nil {
		return fmt.Errorf("read dimensions: %w", err)
	}

	width := binary.LittleEndian.Uint32(buf[:4])
	height := binary.LittleEndian.Uint32(buf[4:])

	println(width, height)

	img, err := NewImage(r, width, height)
	if err != nil {
		return err
	}
	switch ext {
	case ".png":
		return png.Encode(w, img)
	case ".jpg", ".jpeg":
		return jpeg.Encode(w, img, &jpeg.Options{Quality: 100})
	default:
		return errors.New("unknown filetype")
	}
}
