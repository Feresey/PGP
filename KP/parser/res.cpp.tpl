void obj(std::vector<Polygon>& poly, const Object& fp)
{
    std::vector<Polygon> object_poly = {
        {{- range $idx, $_ := .Polygons }}
        {{if not (eq $idx 0)}},{{end}}{{show .}}
        {{- end}}
        };
    for (auto& it : object_poly) {
        float k = fp.radius / {{.MaxCoord}};
        vec3 a = (it.a * k) + fp.center;
        vec3 b = (it.b * k) + fp.center;
        vec3 c = (it.c * k) + fp.center;
        poly.push_back({ a, b, c, fp.color });
    }
}
