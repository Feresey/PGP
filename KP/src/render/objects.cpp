#include "render.cuh"

void dodek(std::vector<Polygon>& poly, const Object& fp)
{
    std::vector<Polygon> object_poly = {
        {{0.475500f, 0.654500f, 0.154500f}, {0.769400f, 0.250000f, -0.154500f}, {0.475500f, 0.154500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, 0.809000f, -0.154500f}, {0.475500f, 0.654500f, 0.154500f}, {0.475500f, 0.154500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, 0.500000f, -0.654500f}, {0.000000f, 0.809000f, -0.154500f}, {0.475500f, 0.154500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, 0.654500f, 0.154500f}, {0.000000f, 0.809000f, -0.154500f}, {0.000000f, 0.500000f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.769400f, 0.250000f, -0.154500f}, {-0.475500f, 0.654500f, 0.154500f}, {0.000000f, 0.500000f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, 0.154500f, -0.654500f}, {-0.769400f, 0.250000f, -0.154500f}, {0.000000f, 0.500000f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.769400f, -0.250000f, 0.154500f}, {-0.769400f, 0.250000f, -0.154500f}, {-0.475500f, 0.154500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, -0.654500f, -0.154500f}, {-0.769400f, -0.250000f, 0.154500f}, {-0.475500f, 0.154500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.293900f, -0.404500f, -0.654500f}, {-0.475500f, -0.654500f, -0.154500f}, {-0.475500f, 0.154500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.809000f, 0.154500f}, {-0.475500f, -0.654500f, -0.154500f}, {-0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.475500f, -0.654500f, -0.154500f}, {0.000000f, -0.809000f, 0.154500f}, {-0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.293900f, -0.404500f, -0.654500f}, {0.475500f, -0.654500f, -0.154500f}, {-0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.769400f, -0.250000f, 0.154500f}, {0.475500f, -0.654500f, -0.154500f}, {0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.769400f, 0.250000f, -0.154500f}, {0.769400f, -0.250000f, 0.154500f}, {0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.475500f, 0.154500f, -0.654500f}, {0.769400f, 0.250000f, -0.154500f}, {0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.475500f, -0.154500f, 0.654500f}, {0.769400f, -0.250000f, 0.154500f}, {0.769400f, 0.250000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.293900f, 0.404500f, 0.654500f}, {0.475500f, -0.154500f, 0.654500f}, {0.769400f, 0.250000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.475500f, 0.654500f, 0.154500f}, {0.293900f, 0.404500f, 0.654500f}, {0.769400f, 0.250000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.293900f, 0.404500f, 0.654500f}, {0.475500f, 0.654500f, 0.154500f}, {0.000000f, 0.809000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.293900f, 0.404500f, 0.654500f}, {0.293900f, 0.404500f, 0.654500f}, {0.000000f, 0.809000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, 0.654500f, 0.154500f}, {-0.293900f, 0.404500f, 0.654500f}, {0.000000f, 0.809000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.293900f, 0.404500f, 0.654500f}, {-0.475500f, 0.654500f, 0.154500f}, {-0.769400f, 0.250000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, -0.154500f, 0.654500f}, {-0.293900f, 0.404500f, 0.654500f}, {-0.769400f, 0.250000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.769400f, -0.250000f, 0.154500f}, {-0.475500f, -0.154500f, 0.654500f}, {-0.769400f, 0.250000f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, -0.154500f, 0.654500f}, {-0.769400f, -0.250000f, 0.154500f}, {-0.475500f, -0.654500f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.500000f, 0.654500f}, {-0.475500f, -0.154500f, 0.654500f}, {-0.475500f, -0.654500f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.809000f, 0.154500f}, {0.000000f, -0.500000f, 0.654500f}, {-0.475500f, -0.654500f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.500000f, 0.654500f}, {0.000000f, -0.809000f, 0.154500f}, {0.475500f, -0.654500f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.475500f, -0.154500f, 0.654500f}, {0.000000f, -0.500000f, 0.654500f}, {0.475500f, -0.654500f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.769400f, -0.250000f, 0.154500f}, {0.475500f, -0.154500f, 0.654500f}, {0.475500f, -0.654500f, -0.154500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.500000f, 0.654500f}, {0.475500f, -0.154500f, 0.654500f}, {0.293900f, 0.404500f, 0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, -0.154500f, 0.654500f}, {0.000000f, -0.500000f, 0.654500f}, {0.293900f, 0.404500f, 0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.293900f, 0.404500f, 0.654500f}, {-0.475500f, -0.154500f, 0.654500f}, {0.293900f, 0.404500f, 0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, 0.500000f, -0.654500f}, {0.475500f, 0.154500f, -0.654500f}, {0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.475500f, 0.154500f, -0.654500f}, {0.000000f, 0.500000f, -0.654500f}, {0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.293900f, -0.404500f, -0.654500f}, {-0.475500f, 0.154500f, -0.654500f}, {0.293900f, -0.404500f, -0.654500f}, {0.0f, 0.0f, 0.0f}}
        };
    for (auto& it : object_poly) {
        float k = fp.radius / 0.8236323572971606f;
        vec3 a = (it.a * k) + fp.center;
        vec3 b = (it.b * k) + fp.center;
        vec3 c = (it.c * k) + fp.center;
        poly.push_back({ a, b, c, fp.color });
    }
}

void icos(std::vector<Polygon>& poly, const Object& fp)
{
    std::vector<Polygon> object_poly = {
        {{0.000000f, 0.500000f, -0.000000f}, {-0.000000f, -0.000000f, -0.500000f}, {-0.026100f, 0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.026100f, -0.447800f}, {-0.026100f, 0.447800f, -0.026100f}, {0.000000f, 0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.000000f, -0.000000f, -0.500000f}, {-0.500000f, -0.000000f, 0.000000f}, {-0.447800f, 0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, 0.026100f, -0.026100f}, {-0.026100f, 0.026100f, -0.447800f}, {-0.000000f, -0.000000f, -0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.500000f, -0.000000f, 0.000000f}, {0.000000f, 0.500000f, -0.000000f}, {-0.026100f, 0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.447800f, -0.026100f}, {-0.447800f, 0.026100f, -0.026100f}, {-0.500000f, -0.000000f, 0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, 0.500000f, -0.000000f}, {-0.500000f, -0.000000f, 0.000000f}, {-0.447800f, 0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, 0.026100f, 0.026100f}, {-0.026100f, 0.447800f, 0.026100f}, {0.000000f, 0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.500000f, -0.000000f, 0.000000f}, {0.000000f, -0.000000f, 0.500000f}, {-0.026100f, 0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.026100f, 0.447800f}, {-0.447800f, 0.026100f, 0.026100f}, {-0.500000f, -0.000000f, 0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.000000f, 0.500000f}, {0.000000f, 0.500000f, -0.000000f}, {-0.026100f, 0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.447800f, 0.026100f}, {-0.026100f, 0.026100f, 0.447800f}, {0.000000f, -0.000000f, 0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, 0.500000f, -0.000000f}, {0.000000f, -0.000000f, 0.500000f}, {0.026100f, 0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.026100f, 0.447800f}, {0.026100f, 0.447800f, 0.026100f}, {0.000000f, 0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.000000f, 0.500000f}, {0.500000f, -0.000000f, -0.000000f}, {0.447800f, 0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, 0.026100f, 0.026100f}, {0.026100f, 0.026100f, 0.447800f}, {0.000000f, -0.000000f, 0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.500000f, -0.000000f, -0.000000f}, {0.000000f, 0.500000f, -0.000000f}, {0.026100f, 0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.447800f, 0.026100f}, {0.447800f, 0.026100f, 0.026100f}, {0.500000f, -0.000000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, 0.500000f, -0.000000f}, {0.500000f, -0.000000f, -0.000000f}, {0.447800f, 0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, 0.026100f, -0.026100f}, {0.026100f, 0.447800f, -0.026100f}, {0.000000f, 0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.500000f, -0.000000f, -0.000000f}, {-0.000000f, -0.000000f, -0.500000f}, {0.026100f, 0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.026100f, -0.447800f}, {0.447800f, 0.026100f, -0.026100f}, {0.500000f, -0.000000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.000000f, -0.000000f, -0.500000f}, {0.000000f, 0.500000f, -0.000000f}, {0.026100f, 0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.447800f, -0.026100f}, {0.026100f, 0.026100f, -0.447800f}, {-0.000000f, -0.000000f, -0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.500000f, -0.000000f}, {-0.500000f, -0.000000f, 0.000000f}, {-0.447800f, -0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, -0.026100f, -0.026100f}, {-0.026100f, -0.447800f, -0.026100f}, {0.000000f, -0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.500000f, -0.000000f, 0.000000f}, {-0.000000f, -0.000000f, -0.500000f}, {-0.026100f, -0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.026100f, -0.447800f}, {-0.447800f, -0.026100f, -0.026100f}, {-0.500000f, -0.000000f, 0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.000000f, -0.000000f, -0.500000f}, {0.000000f, -0.500000f, -0.000000f}, {-0.026100f, -0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.447800f, -0.026100f}, {-0.026100f, -0.026100f, -0.447800f}, {-0.000000f, -0.000000f, -0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.500000f, -0.000000f}, {0.000000f, -0.000000f, 0.500000f}, {-0.026100f, -0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.026100f, 0.447800f}, {-0.026100f, -0.447800f, 0.026100f}, {0.000000f, -0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.000000f, 0.500000f}, {-0.500000f, -0.000000f, 0.000000f}, {-0.447800f, -0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, -0.026100f, 0.026100f}, {-0.026100f, -0.026100f, 0.447800f}, {0.000000f, -0.000000f, 0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.500000f, -0.000000f, 0.000000f}, {0.000000f, -0.500000f, -0.000000f}, {-0.026100f, -0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.447800f, 0.026100f}, {-0.447800f, -0.026100f, 0.026100f}, {-0.500000f, -0.000000f, 0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.500000f, -0.000000f}, {0.500000f, -0.000000f, -0.000000f}, {0.447800f, -0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, -0.026100f, 0.026100f}, {0.026100f, -0.447800f, 0.026100f}, {0.000000f, -0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.500000f, -0.000000f, -0.000000f}, {0.000000f, -0.000000f, 0.500000f}, {0.026100f, -0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.026100f, 0.447800f}, {0.447800f, -0.026100f, 0.026100f}, {0.500000f, -0.000000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.000000f, 0.500000f}, {0.000000f, -0.500000f, -0.000000f}, {0.026100f, -0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.447800f, 0.026100f}, {0.026100f, -0.026100f, 0.447800f}, {0.000000f, -0.000000f, 0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.500000f, -0.000000f}, {-0.000000f, -0.000000f, -0.500000f}, {0.026100f, -0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.026100f, -0.447800f}, {0.026100f, -0.447800f, -0.026100f}, {0.000000f, -0.500000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.000000f, -0.000000f, -0.500000f}, {0.500000f, -0.000000f, -0.000000f}, {0.447800f, -0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, -0.026100f, -0.026100f}, {0.026100f, -0.026100f, -0.447800f}, {-0.000000f, -0.000000f, -0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.500000f, -0.000000f, -0.000000f}, {0.000000f, -0.500000f, -0.000000f}, {0.026100f, -0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.447800f, -0.026100f}, {0.447800f, -0.026100f, -0.026100f}, {0.500000f, -0.000000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.447800f, -0.026100f}, {-0.026100f, 0.026100f, -0.447800f}, {-0.021300f, 0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.021300f, -0.443000f}, {-0.021300f, 0.443000f, -0.021300f}, {-0.026100f, 0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.026100f, -0.447800f}, {-0.447800f, 0.026100f, -0.026100f}, {-0.443000f, 0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, 0.021300f, -0.021300f}, {-0.021300f, 0.021300f, -0.443000f}, {-0.026100f, 0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, 0.026100f, -0.026100f}, {-0.026100f, 0.447800f, -0.026100f}, {-0.021300f, 0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.443000f, -0.021300f}, {-0.443000f, 0.021300f, -0.021300f}, {-0.447800f, 0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.447800f, 0.026100f}, {-0.447800f, 0.026100f, 0.026100f}, {-0.443000f, 0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, 0.021300f, 0.021300f}, {-0.021300f, 0.443000f, 0.021300f}, {-0.026100f, 0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, 0.026100f, 0.026100f}, {-0.026100f, 0.026100f, 0.447800f}, {-0.021300f, 0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.021300f, 0.443000f}, {-0.443000f, 0.021300f, 0.021300f}, {-0.447800f, 0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, 0.026100f, 0.447800f}, {-0.026100f, 0.447800f, 0.026100f}, {-0.021300f, 0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.443000f, 0.021300f}, {-0.021300f, 0.021300f, 0.443000f}, {-0.026100f, 0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.447800f, 0.026100f}, {0.026100f, 0.026100f, 0.447800f}, {0.021300f, 0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.021300f, 0.443000f}, {0.021300f, 0.443000f, 0.021300f}, {0.026100f, 0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.026100f, 0.447800f}, {0.447800f, 0.026100f, 0.026100f}, {0.443000f, 0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, 0.021300f, 0.021300f}, {0.021300f, 0.021300f, 0.443000f}, {0.026100f, 0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, 0.026100f, 0.026100f}, {0.026100f, 0.447800f, 0.026100f}, {0.021300f, 0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.443000f, 0.021300f}, {0.443000f, 0.021300f, 0.021300f}, {0.447800f, 0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.447800f, -0.026100f}, {0.447800f, 0.026100f, -0.026100f}, {0.443000f, 0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, 0.021300f, -0.021300f}, {0.021300f, 0.443000f, -0.021300f}, {0.026100f, 0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, 0.026100f, -0.026100f}, {0.026100f, 0.026100f, -0.447800f}, {0.021300f, 0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.021300f, -0.443000f}, {0.443000f, 0.021300f, -0.021300f}, {0.447800f, 0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, 0.026100f, -0.447800f}, {0.026100f, 0.447800f, -0.026100f}, {0.021300f, 0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.443000f, -0.021300f}, {0.021300f, 0.021300f, -0.443000f}, {0.026100f, 0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.447800f, -0.026100f}, {-0.447800f, -0.026100f, -0.026100f}, {-0.443000f, -0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, -0.021300f, -0.021300f}, {-0.021300f, -0.443000f, -0.021300f}, {-0.026100f, -0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, -0.026100f, -0.026100f}, {-0.026100f, -0.026100f, -0.447800f}, {-0.021300f, -0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.021300f, -0.443000f}, {-0.443000f, -0.021300f, -0.021300f}, {-0.447800f, -0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.026100f, -0.447800f}, {-0.026100f, -0.447800f, -0.026100f}, {-0.021300f, -0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.443000f, -0.021300f}, {-0.021300f, -0.021300f, -0.443000f}, {-0.026100f, -0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.447800f, 0.026100f}, {-0.026100f, -0.026100f, 0.447800f}, {-0.021300f, -0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.021300f, 0.443000f}, {-0.021300f, -0.443000f, 0.021300f}, {-0.026100f, -0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.026100f, -0.026100f, 0.447800f}, {-0.447800f, -0.026100f, 0.026100f}, {-0.443000f, -0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, -0.021300f, 0.021300f}, {-0.021300f, -0.021300f, 0.443000f}, {-0.026100f, -0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.447800f, -0.026100f, 0.026100f}, {-0.026100f, -0.447800f, 0.026100f}, {-0.021300f, -0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.443000f, 0.021300f}, {-0.443000f, -0.021300f, 0.021300f}, {-0.447800f, -0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.447800f, 0.026100f}, {0.447800f, -0.026100f, 0.026100f}, {0.443000f, -0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, -0.021300f, 0.021300f}, {0.021300f, -0.443000f, 0.021300f}, {0.026100f, -0.447800f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, -0.026100f, 0.026100f}, {0.026100f, -0.026100f, 0.447800f}, {0.021300f, -0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.021300f, 0.443000f}, {0.443000f, -0.021300f, 0.021300f}, {0.447800f, -0.026100f, 0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.026100f, 0.447800f}, {0.026100f, -0.447800f, 0.026100f}, {0.021300f, -0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.443000f, 0.021300f}, {0.021300f, -0.021300f, 0.443000f}, {0.026100f, -0.026100f, 0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.447800f, -0.026100f}, {0.026100f, -0.026100f, -0.447800f}, {0.021300f, -0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.021300f, -0.443000f}, {0.021300f, -0.443000f, -0.021300f}, {0.026100f, -0.447800f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.026100f, -0.026100f, -0.447800f}, {0.447800f, -0.026100f, -0.026100f}, {0.443000f, -0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, -0.021300f, -0.021300f}, {0.021300f, -0.021300f, -0.443000f}, {0.026100f, -0.026100f, -0.447800f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.447800f, -0.026100f, -0.026100f}, {0.026100f, -0.447800f, -0.026100f}, {0.021300f, -0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.443000f, -0.021300f}, {0.443000f, -0.021300f, -0.021300f}, {0.447800f, -0.026100f, -0.026100f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.443000f, -0.021300f}, {-0.021300f, 0.443000f, -0.021300f}, {-0.021300f, 0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.021300f, -0.443000f}, {0.021300f, 0.021300f, -0.443000f}, {0.021300f, 0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, -0.021300f, -0.021300f}, {0.443000f, 0.021300f, -0.021300f}, {0.021300f, 0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.021300f, -0.443000f}, {0.021300f, -0.021300f, -0.443000f}, {0.443000f, -0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.021300f, -0.443000f}, {-0.021300f, 0.021300f, -0.443000f}, {-0.443000f, 0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, 0.021300f, -0.021300f}, {-0.443000f, -0.021300f, -0.021300f}, {-0.021300f, -0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.021300f, -0.443000f}, {-0.021300f, -0.021300f, -0.443000f}, {-0.021300f, -0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.443000f, -0.021300f}, {0.021300f, -0.443000f, -0.021300f}, {0.021300f, -0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.021300f, -0.443000f}, {0.021300f, -0.021300f, -0.443000f}, {0.021300f, 0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.021300f, -0.443000f}, {-0.021300f, 0.021300f, -0.443000f}, {-0.021300f, -0.021300f, -0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, 0.021300f, 0.021300f}, {-0.443000f, 0.021300f, -0.021300f}, {-0.021300f, 0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.443000f, -0.021300f}, {-0.021300f, 0.443000f, 0.021300f}, {-0.443000f, 0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, -0.021300f, 0.021300f}, {-0.443000f, 0.021300f, 0.021300f}, {-0.021300f, 0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.021300f, 0.443000f}, {-0.021300f, -0.021300f, 0.443000f}, {-0.443000f, -0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.443000f, 0.021300f}, {-0.021300f, -0.443000f, -0.021300f}, {-0.443000f, -0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, -0.021300f, -0.021300f}, {-0.443000f, -0.021300f, 0.021300f}, {-0.021300f, -0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, -0.021300f, 0.021300f}, {-0.443000f, -0.021300f, -0.021300f}, {-0.443000f, 0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.443000f, 0.021300f, -0.021300f}, {-0.443000f, 0.021300f, 0.021300f}, {-0.443000f, -0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.021300f, 0.443000f}, {-0.021300f, 0.021300f, 0.443000f}, {-0.021300f, 0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.443000f, 0.021300f}, {0.021300f, 0.443000f, 0.021300f}, {0.021300f, 0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.021300f, 0.443000f}, {0.021300f, 0.021300f, 0.443000f}, {0.443000f, 0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, 0.021300f, 0.021300f}, {0.443000f, -0.021300f, 0.021300f}, {0.021300f, -0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.021300f, 0.443000f}, {-0.021300f, -0.021300f, 0.443000f}, {-0.021300f, 0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.021300f, 0.443000f}, {0.021300f, 0.021300f, 0.443000f}, {0.021300f, -0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.443000f, 0.021300f}, {-0.021300f, -0.443000f, 0.021300f}, {-0.021300f, -0.021300f, 0.443000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.021300f, 0.443000f}, {0.021300f, -0.021300f, 0.443000f}, {0.021300f, -0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.443000f, -0.021300f}, {0.021300f, -0.443000f, 0.021300f}, {0.443000f, -0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, -0.021300f, 0.021300f}, {0.443000f, -0.021300f, -0.021300f}, {0.021300f, -0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, -0.443000f, 0.021300f}, {0.021300f, -0.443000f, -0.021300f}, {-0.021300f, -0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, -0.443000f, -0.021300f}, {-0.021300f, -0.443000f, 0.021300f}, {0.021300f, -0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, 0.021300f, 0.021300f}, {0.443000f, 0.021300f, -0.021300f}, {0.443000f, -0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, -0.021300f, -0.021300f}, {0.443000f, -0.021300f, 0.021300f}, {0.443000f, 0.021300f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.443000f, 0.021300f, -0.021300f}, {0.443000f, 0.021300f, 0.021300f}, {0.021300f, 0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.443000f, 0.021300f}, {0.021300f, 0.443000f, -0.021300f}, {0.443000f, 0.021300f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.021300f, 0.443000f, 0.021300f}, {-0.021300f, 0.443000f, -0.021300f}, {0.021300f, 0.443000f, -0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.021300f, 0.443000f, -0.021300f}, {0.021300f, 0.443000f, 0.021300f}, {-0.021300f, 0.443000f, 0.021300f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.019600f, 0.840800f, 0.448600f}, {0.828700f, 0.531800f, -0.051400f}, {0.019600f, 0.840800f, -0.551400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.828700f, 0.531800f, -0.051400f}, {0.828700f, -0.468200f, -0.051400f}, {0.019600f, -0.777200f, 0.448600f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.019600f, 0.840800f, 0.448600f}, {0.828700f, 0.531800f, -0.051400f}, {0.019600f, -0.777200f, 0.448600f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, 0.757600f}, {0.019600f, 0.840800f, 0.448600f}, {0.019600f, -0.777200f, 0.448600f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.789400f, 0.531800f, -0.051400f}, {-0.480400f, 0.031800f, -0.860500f}, {0.019600f, -0.777200f, -0.551400f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, 0.757600f}, {-0.789400f, 0.531800f, -0.051400f}, {0.019600f, -0.777200f, -0.551400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.019600f, -0.777200f, 0.448600f}, {-0.480400f, 0.031800f, 0.757600f}, {0.019600f, -0.777200f, -0.551400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.019600f, -0.777200f, 0.448600f}, {-0.480400f, 0.031800f, 0.757600f}, {-0.789400f, -0.468200f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, 0.757600f}, {0.019600f, -0.777200f, 0.448600f}, {0.519700f, 0.031800f, 0.757600f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.519700f, 0.031800f, 0.757600f}, {0.019600f, -0.777200f, 0.448600f}, {0.828700f, -0.468200f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.828700f, -0.468200f, -0.051400f}, {0.019600f, -0.777200f, 0.448600f}, {0.019600f, -0.777200f, -0.551400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.019600f, -0.777200f, 0.448600f}, {-0.789400f, -0.468200f, -0.051400f}, {0.019600f, -0.777200f, -0.551400f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, -0.860500f}, {0.019600f, -0.777200f, -0.551400f}, {-0.789400f, -0.468200f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.519700f, 0.031800f, -0.860500f}, {0.019600f, -0.777200f, -0.551400f}, {-0.480400f, 0.031800f, -0.860500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.828700f, -0.468200f, -0.051400f}, {0.019600f, -0.777200f, -0.551400f}, {0.519700f, 0.031800f, -0.860500f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.828700f, -0.468200f, -0.051400f}, {0.519700f, 0.031800f, -0.860500f}, {0.828700f, 0.531800f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.519700f, 0.031800f, -0.860500f}, {0.019600f, 0.840800f, -0.551400f}, {0.828700f, 0.531800f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, -0.860500f}, {0.019600f, 0.840800f, -0.551400f}, {0.519700f, 0.031800f, -0.860500f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, -0.860500f}, {-0.789400f, 0.531800f, -0.051400f}, {0.019600f, 0.840800f, -0.551400f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, -0.860500f}, {-0.789400f, -0.468200f, -0.051400f}, {-0.789400f, 0.531800f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.789400f, -0.468200f, -0.051400f}, {-0.480400f, 0.031800f, 0.757600f}, {-0.789400f, 0.531800f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.480400f, 0.031800f, 0.757600f}, {0.019600f, 0.840800f, 0.448600f}, {-0.789400f, 0.531800f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.019600f, 0.840800f, 0.448600f}, {-0.480400f, 0.031800f, 0.757600f}, {0.519700f, 0.031800f, 0.757600f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.519700f, 0.031800f, 0.757600f}, {0.828700f, 0.531800f, -0.051400f}, {0.019600f, 0.840800f, 0.448600f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.519700f, 0.031800f, 0.757600f}, {0.828700f, -0.468200f, -0.051400f}, {0.828700f, 0.531800f, -0.051400f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.019600f, 0.840800f, -0.551400f}, {-0.789400f, 0.531800f, -0.051400f}, {0.019600f, 0.840800f, 0.448600f}, {0.0f, 0.0f, 0.0f}}
        };
    for (auto& it : object_poly) {
        float k = fp.radius / 1.0057631593739496f;
        vec3 a = (it.a * k) + fp.center;
        vec3 b = (it.b * k) + fp.center;
        vec3 c = (it.c * k) + fp.center;
        poly.push_back({ a, b, c, fp.color });
    }
}

void octa(std::vector<Polygon>& poly, const Object& fp)
{
    std::vector<Polygon> object_poly = {
        {{-0.500000f, -0.000000f, 0.000000f}, {0.000000f, 0.500000f, -0.000000f}, {-0.000000f, -0.000000f, -0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.000000f, 0.500000f}, {0.000000f, 0.500000f, -0.000000f}, {-0.500000f, -0.000000f, 0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.500000f, -0.000000f, -0.000000f}, {0.000000f, 0.500000f, -0.000000f}, {0.000000f, -0.000000f, 0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.000000f, -0.000000f, -0.500000f}, {0.000000f, 0.500000f, -0.000000f}, {0.500000f, -0.000000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.000000f, -0.000000f, -0.500000f}, {0.000000f, -0.500000f, -0.000000f}, {-0.500000f, -0.000000f, 0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{-0.500000f, -0.000000f, 0.000000f}, {0.000000f, -0.500000f, -0.000000f}, {0.000000f, -0.000000f, 0.500000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.000000f, -0.000000f, 0.500000f}, {0.000000f, -0.500000f, -0.000000f}, {0.500000f, -0.000000f, -0.000000f}, {0.0f, 0.0f, 0.0f}}
        ,{{0.500000f, -0.000000f, -0.000000f}, {0.000000f, -0.500000f, -0.000000f}, {-0.000000f, -0.000000f, -0.500000f}, {0.0f, 0.0f, 0.0f}}
        };
    for (auto& it : object_poly) {
        float k = fp.radius / 0.5f;
        vec3 a = (it.a * k) + fp.center;
        vec3 b = (it.b * k) + fp.center;
        vec3 c = (it.c * k) + fp.center;
        poly.push_back({ a, b, c, fp.color });
    }
}