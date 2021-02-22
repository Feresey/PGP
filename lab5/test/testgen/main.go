package main

import (
	"bufio"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"log"
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
	pflag.StringVarP(&inName, "out-in", "i", "in.hex", "filename for input array")
	pflag.StringVarP(&outName, "out-want", "w", "want.hex", "filename for sorted result array")
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

	log.Println("generate array")
	for idx := range arr {
		num, err := rand.Int(rand.Reader, big.NewInt(1<<31))
		if err != nil {
			panic(err)
		}
		arr[idx] = int(num.Int64())
	}

	log.Println("write generated")
	if err := writeArr(outIn, arr, true); err != nil {
		panic(err)
	}
	log.Println("sort")
	sort.Ints(arr)
	log.Println("write result")
	if err := writeArr(outWant, arr, false); err != nil {
		panic(err)
	}
}

func writeArr(w io.Writer, arr []int, writeLen bool) error {
	ww := bufio.NewWriterSize(w, 1<<20)
	defer ww.Flush()

	var buf [4]byte

	swap := func(i int) int {
		binary.LittleEndian.PutUint32(buf[:], uint32(i))
		return int(binary.BigEndian.Uint32(buf[:]))
	}

	if writeLen {
		fmt.Fprintf(ww, "%08X ", swap(len(arr)))
	}

	for idx, elem := range arr {
		fmt.Fprintf(ww, "%08X", swap(elem))
		if idx+1 != len(arr) {
			_ = ww.WriteByte(' ')
		}
	}
	_ = ww.WriteByte('\n')

	return nil
}
