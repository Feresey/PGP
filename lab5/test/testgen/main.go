package main

import (
	"bufio"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math/big"
	"os"
	"sort"

	"github.com/spf13/pflag"
)

func main() {
	var (
		size    int
		inName  string
		outName string
	)
	pflag.IntVarP(&size, "size", "s", 1024, "length of result array")
	pflag.StringVarP(&inName, "out-in", "i", "in", "filename for input array")
	pflag.StringVarP(&outName, "out-want", "w", "want", "filename for sorted result array")
	pflag.Parse()

	outIn, err := os.Create(inName)
	if err != nil {
		panic(err)
	}
	defer outIn.Close()

	outWant, err := os.Create(outName)
	if err != nil {
		panic(err)
	}
	defer outWant.Close()

	arr := make([]int, size)

	println("generate array")
	for idx := range arr {
		num, err := rand.Int(rand.Reader, big.NewInt(1<<31))
		if err != nil {
			panic(err)
		}
		arr[idx] = int(num.Int64())
	}

	println("write generated")
	if err := writeArr(outIn, arr, true); err != nil {
		panic(err)
	}
	println("sort")
	sort.Ints(arr)
	println("write result")
	if err := writeArr(outWant, arr, false); err != nil {
		panic(err)
	}
}

func writeArr(w io.Writer, arr []int, writeLen bool) error {
	ww := bufio.NewWriterSize(w, 1<<20)
	defer ww.Flush()

	var (
		buf [4]byte
		enc = hex.NewEncoder(ww)
	)

	if writeLen {
		binary.LittleEndian.PutUint32(buf[:], uint32(len(arr)))
		if _, err := enc.Write(buf[:]); err != nil {
			return err
		}
		_, _ = ww.Write([]byte(" "))
	}

	for idx, elem := range arr {
		binary.LittleEndian.PutUint32(buf[:], uint32(elem))
		if _, err := enc.Write(buf[:]); err != nil {
			return err
		}
		if idx+1 != len(arr) {
			_, _ = ww.Write([]byte(" "))
		}
	}
	_, _ = ww.Write([]byte("\n"))

	return nil
}
