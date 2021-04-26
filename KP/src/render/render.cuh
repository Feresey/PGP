#ifndef RENDER_CUH
#define RENDER_CUH

#include "scene.hpp"
#include "vec/vec3.hpp"

enum ComputeMode {
    CUDA,
    OPEN_MP
};

struct OpenMPRenderer {
    Scene scene;
    std::vector<Polygon> poly;

    int ssaa_rate;
    int render_w, render_h;
    int render_size;

    std::vector<uchar4> data_render, data_ssaa;

    const uchar4* data() const;
    void mpi_bcast_poly();

    OpenMPRenderer(const Scene& scene);
    void Render(int frame);
};

struct CUDARenderer {
    Scene scene;
    std::vector<Polygon> poly;

    int ssaa_rate;
    int render_w, render_h;
    int render_size;

    std::vector<uchar4> data_render, data_ssaa;

    Polygon* dev_poly;
    Light* dev_lights;
    uchar4* dev_render;
    uchar4* dev_ssaa;

    const uchar4* data() const;
    void mpi_bcast_poly();

    CUDARenderer(const Scene& scene);
    void retarded(const Scene& scene);
    ~CUDARenderer();
    void Render(int frame);
};

__host__ __device__ uchar4 ray(
    const vec3& pos, const vec3& dir,
    const Polygon* scene_trigs, int ntrigs,
    const Light* lights, int nlights);

std::vector<Polygon> polygons(const Scene& scene);
std::pair<vec3, vec3> cum_view(const Scene& scene, int frame);

#endif
