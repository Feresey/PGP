#include <mpi.h>
#include <omp.h>

#include "helpers.hpp"
#include "render.cuh"
#include "ssaa.hpp"

Renderer::Renderer(Scene scene)
    : scene(scene)
    , poly(polygons(scene))
    , ssaa_rate(2)
    , render_w(ssaa_rate * scene.w)
    , render_h(ssaa_rate * scene.h)
    , render_size(render_w * render_h)
    , data_render(size_t(render_size))
    , data_ssaa(size_t(scene.w * scene.h))
{
}

void octa(std::vector<Polygon>& poly, const Object& fp);
void dodek(std::vector<Polygon>& poly, const Object& fp);
void icos(std::vector<Polygon>& poly, const Object& fp);

std::vector<Polygon> polygons(const Scene& scene)
{
    std::vector<Polygon> poly;
    octa(poly, scene.octahedron);
    dodek(poly, scene.dodecahedron);
    icos(poly, scene.icosahedron);
    auto fp = scene.floor;
    std::cerr << fp << std::endl;
    poly.push_back({ fp.c, fp.b, fp.a, fp.color });
    poly.push_back({ fp.a, fp.d, fp.c, fp.color });
    return poly;
}

const uchar4* Renderer::data() const
{
    return data_ssaa.data();
}

std::pair<vec3, vec3> Renderer::cum_view(int frame) const
{
    const Camera& c = scene.cum_center;
    const Camera& n = scene.cum_dir;
    const float t = 0.01f * float(frame);

    float rc = c.r0 + c.ar * sinf(c.wr * t + c.pr);
    float zc = c.z0 + c.ar * sinf(c.wz * t + c.pz);
    float phic = c.phi0 + c.wphi * t;

    float rn = n.r0 + n.ar * sinf(n.wr * t + n.pr);
    float zn = n.z0 + n.ar * sinf(n.wz * t + n.pz);
    float phin = n.phi0 + n.wphi * t;

    return {
        vec3(rc * cosf(phic), rc * sinf(phic), zc),
        vec3(rn * cosf(phin), rn * sinf(phin), zn)
    };
}

Renderer* NewRenderer(ComputeMode mode, Scene scene)
{
    switch (mode) {
    default:
    case CUDA:
        return new CUDARenderer(scene);
    case OPEN_MP:
        return new OpenMPRenderer(scene);
    }
}

OpenMPRenderer::OpenMPRenderer(Scene scene)
    : Renderer(scene)
{
}

void Renderer::mpi_bcast_poly()
{
    bcast_bytes(poly.data(), int(poly.size() * sizeof(Polygon)));
}

void OpenMPRenderer::Render(int frame)
{
    const float dw = 2.0f / (float(render_w) - 1.0f);
    const float dh = 2.0f / (float(render_h) - 1.0f);
    const float z = 1.0f / tanf((scene.angle * M_PIf32) / 360.0f);
    std::pair<vec3, vec3> p = cum_view(frame);

    vec3 bz = (p.second - p.first).normalize();
    vec3 bx = vec3::cross_product(bz, vec3(0.0f, 0.0f, 1.0f)).normalize();
    vec3 by = vec3::cross_product(bx, bz).normalize();

#pragma omp parallel for
    for (int pix = 0; pix < render_w * render_h; ++pix) {
        int i = pix % render_w;
        int j = pix / render_w;
        const vec3 v(-1.0f + dw * float(i), (-1.0f + dh * float(j)) * float(render_h) / float(render_w), z);
        vec3 dir = vec3(
            bx.x * v.x + by.x * v.y + bz.x * v.z,
            bx.y * v.x + by.y * v.y + bz.y * v.z,
            bx.z * v.x + by.z * v.y + bz.z * v.z)
                       .normalize();
        data_render[size_t((render_h - 1 - j) * render_w + i)] = ray(
            p.first, dir,
            poly.data(), int(poly.size()),
            scene.lights.data(), int(scene.lights.size()));
    }

    ssaa_omp(data_ssaa.data(), data_render.data(), scene.w, scene.h, render_w, render_h);
}
