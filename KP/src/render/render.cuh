#ifndef RENDER_CUH
#define RENDER_CUH

#include "scene.hpp"
#include "vec/vec3.hpp"


enum ComputeMode {
    CUDA,
    OPEN_MP
};

class Renderer {
protected:
    const Scene scene;
    std::vector<Polygon> poly;

    const int ssaa_rate;
    const int render_w, render_h;
    const int render_size;

    std::vector<uchar4> data_render, data_ssaa;

    std::pair<vec3, vec3> cum_view(int frame) const;
    Renderer(Scene scene);

public:
    void mpi_bcast_poly();
    const uchar4* data() const;
    virtual void Render(int frame) = 0;
    virtual ~Renderer() = default;
};

class OpenMPRenderer : public Renderer {
public:
    OpenMPRenderer(Scene scene);
    void Render(int frame) override;
};

class CUDARenderer : public Renderer {
    Polygon* dev_poly;
    Light* dev_lights;
    uchar4* dev_render;
    uchar4* dev_ssaa;

public:
    CUDARenderer(Scene scene);
    ~CUDARenderer();
    void Render(int frame) override;
};

__host__ __device__ uchar4 ray(
    const vec3& pos, const vec3& dir,
    const Polygon* scene_trigs, int ntrigs,
    const Light* lights, int nlights);

std::vector<Polygon> polygons(const Scene& scene);

Renderer* NewRenderer(ComputeMode mode, Scene scene);

#endif
