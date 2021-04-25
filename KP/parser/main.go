package main

import (
	"bufio"
	_ "embed"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"text/template"
)

type vec3 struct {
	x, y, z float64
}

func (v vec3) String() string { return fmt.Sprintf("{%f, %f, %f}", v.x, v.y, v.z) }

func (v vec3) Len() float64 { return math.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z) }

type Polygon struct {
	a, b, c vec3
}

type Object struct {
	Vertices []vec3
	Polygons []Polygon
	MaxCoord float64
}

func readObj(r io.Reader) (*Object, error) {
	var obj Object
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		if len(scanner.Text()) == 0 {
			continue
		}
		words := strings.Split(scanner.Text(), " ")
		switch words[0] {
		case "v":
			var v vec3
			v.x, _ = strconv.ParseFloat(words[2], 32)
			v.y, _ = strconv.ParseFloat(words[3], 32)
			v.z, _ = strconv.ParseFloat(words[4], 32)
			obj.Vertices = append(obj.Vertices, v)
		case "f":
			getInt := func(s string) int {
				i, _ := strconv.Atoi(strings.Split(s, "/")[0])
				return i
			}
			ia := getInt(words[1])
			ib := getInt(words[2])
			ic := getInt(words[3])

			a := obj.Vertices[ia-1]
			b := obj.Vertices[ib-1]
			c := obj.Vertices[ic-1]
			mx := func(v vec3) {
				ln := v.Len()
				if obj.MaxCoord < ln {
					obj.MaxCoord = ln
				}
			}
			mx(a)
			mx(b)
			mx(c)
			obj.Polygons = append(obj.Polygons, Polygon{a, b, c})
		}
	}
	return &obj, scanner.Err()
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

//go:embed res.cpp.tpl
var tpl string

func main() {
	file, err := os.Open(os.Args[1])
	must(err)
	defer file.Close()

	obj, err := readObj(file)
	must(err)

	err = template.Must(
		template.New("").
			Funcs(template.FuncMap{
				"show": func(p Polygon) string { return fmt.Sprintf("{%s, %s, %s}", p.a, p.b, p.c) },
			}).
			Parse(tpl),
	).Execute(os.Stdout, obj)
	must(err)
}
