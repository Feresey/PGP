package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"
)

func encode(r io.Reader, w io.Writer) error {
	var (
		img image.Image
		err error
	)
	img, _, err = image.Decode(r)
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
		h:   int(h),
		w:   int(w),
		buf: make([][]color.RGBA, h),
	}

	fmt.Printf("w is %d. h is %d.\n", res.w, res.h)

	buf := make([]byte, 4)
	r = bufio.NewReaderSize(r, 1<<20)
	for y := 0; y < res.h; y++ {
		res.buf[y] = make([]color.RGBA, res.w)
		for x := 0; x < res.w; x++ {
			_, err := r.Read(buf)
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

func decode(r io.Reader, w io.Writer, ext string) error {
	buf := make([]byte, 8)
	_, err := r.Read(buf)
	if err != nil {
		return fmt.Errorf("read dismensions: %w", err)
	}

	width := binary.LittleEndian.Uint32(buf[:4])
	height := binary.LittleEndian.Uint32(buf[4:])

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

func main() {
	src := flag.String("i", "", "path to input file")
	dst := flag.String("o", "", "path to output file")
	mode := flag.String("mode", "encode", "convert mode")
	flag.Parse()

	in, err := os.Open(*src)
	if err != nil {
		panic(err)
	}
	defer in.Close()

	out, err := os.Create(*dst)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	switch *mode {
	case "encode":
		if err := encode(in, out); err != nil {
			panic(err)
		}
	case "decode":
		if err := decode(in, out, filepath.Ext(*dst)); err != nil {
			panic(err)
		}
	default:
		panic("unknown mode")
	}
}
