package main

import (
	"bufio"
	"io"
	"math"
	"strconv"
	"strings"
)

type vec3 struct {
	x, y, z float64
}

func (v vec3) Len() float64 { return math.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z) }

type Polygon struct {
	a, b, c vec3
}

type Object struct {
	Vertices []vec3
	Polygons []Polygon
	maxCoord float64
}

func readObj(r io.Reader) (*Object, error) {
	var res Object
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
			res.Vertices = append(res.Vertices, v)
		case "f":
			getInt := func(s string) int {
				i, _ := strconv.Atoi(strings.Split(s, "/")[0])
				return i
			}
			ia := getInt(words[1])
			ib := getInt(words[2])
			ic := getInt(words[3])

			a := res.Vertices[ia]
			b := res.Vertices[ib]
			c := res.Vertices[ic]
			mx := func(v vec3) {
				ln := v.Len()
				if res.maxCoord < ln {
					res.maxCoord = ln
				}
			}
			mx(a)
			mx(b)
			mx(c)
			res.Polygons = append(res.Polygons, Polygon{a, b, c})
		}
	}
	return nil, scanner.Err()
}

// void import_obj_to_scene(std::vector<Trig>& scene_trigs,
//     const std::string& filepath, const Object& fp)
// {element
//     std::ifstream is(filepath);
//     if (!is) {
//         std::string desc = "can't open " + filepath;
//         FAILF(desc);
//     }
//     float r = 0;
//     std::vector<vec3> vertices;
//     std::vector<Trig> figure_trigs;
//     std::string line;
//     while (std::getline(is, line)) {
//         std::vector<std::string> buffer = split_string(line, ' ');
//         if (line.empty()) {
//             continue;
//         } else if (buffer[0] == "v") {
//             float x = std::stod(buffer[2]);
//             float y = std::stod(buffer[3]);
//             float z = std::stod(buffer[4]);

//             vertices.push_back({ x, y, z });
//         } else if (buffer[0] == "f") {
//             std::vector<std::string> indexes = split_string(buffer[1], '/');
//             vec3 a = vertices[std::stoi(indexes[0]) - 1];
//             indexes = split_string(buffer[2], '/');
//             vec3 b = vertices[std::stoi(indexes[0]) - 1];
//             indexes = split_string(buffer[3], '/');
//             vec3 c = vertices[std::stoi(indexes[0]) - 1];

//             r = std::max(r, a.len());
//             r = std::max(r, b.len());
//             r = std::max(r, c.len());

//             figure_trigs.push_back(Trig { a, b, c, fp.color });
//         }
//     }
//     for (auto& it : figure_trigs) {
//         float k = fp.radius / r;
//         vec3 a = (it.a * k) + fp.center;
//         vec3 b = (it.b * k) + fp.center;
//         vec3 c = (it.c * k) + fp.center;
//         scene_trigs.push_back({ a, b, c, it.color });
//     }
// }

func main() {
}
