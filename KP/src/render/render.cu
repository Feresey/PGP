#include "mat/mat3.hpp"
#include "render.cuh"
#include "scene.hpp"
#include "ssaa.hpp"
#include "vec/vec3.hpp"

__host__ __device__ void triangle_intersection(
    const vec3& origin, const vec3& dir,
    const Polygon& trig, float* t,
    float* u, float* v)
{
    vec3 e1 = trig.b - trig.a;
    vec3 e2 = trig.c - trig.a;

    mat3 m(-dir.x, e1.x, e2.x, -dir.y, e1.y, e2.y, -dir.z, e1.z, e2.z);
    vec3 tmp = m.inverse() * (origin - trig.a);

    *t = tmp.x;
    *u = tmp.y;
    *v = tmp.z;
}

__host__ __device__ bool shadow_ray_hit(
    const vec3& origin, const vec3& dir,
    const Polygon* scene_trigs, int ntrigs,
    float* hit_t)
{
    float t_min = 1 / 0.;
    bool hit = false;
    for (int i = 0; i < ntrigs; ++i) {
        auto trig = scene_trigs[i];
        float t, u, v;
        triangle_intersection(origin, dir, trig, &t, &u, &v);
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0) {
            if (t < t_min) {
                t_min = t;
            }
            hit = true;
        }
    }
    *hit_t = t_min;
    return hit;
}

__host__ __device__ vec3 phong_model(
    const vec3& pos, const vec3& dir,
    const Polygon& trig,
    const Polygon* scene_trigs, int ntrigs,
    const Light* lights, int nlights)
{
    constexpr float intensity = 5.f;
    constexpr float ka = 0.1f, kd = 0.6f, ks = 0.5f;

    vec3 normal = vec3::cross_product((trig.b - trig.a), (trig.c - trig.a)).normalize();

    vec3 ambient { ka, ka, ka };
    vec3 diffuse { 0., 0., 0. };
    vec3 specular { 0., 0., 0. };

    for (int i = 0; i < nlights; ++i) {
        vec3 light_pos = lights[i].pos;
        vec3 L = light_pos - pos;
        float d = L.len();
        L = L.normalize();

        float hit_t = 0.0;
        if (shadow_ray_hit(light_pos, -L, scene_trigs, ntrigs,
                &hit_t)
            && (hit_t > d || (hit_t > d || (d - hit_t < 0.0005f)))) {
            float k = intensity / (d + 0.001f);
            diffuse += lights[i].color * max(kd * k * vec3::dot_product(L, normal), 0.0f);

            vec3 R = vec3::reflect(-L, normal).normalize();
            vec3 S = -dir;
            specular += lights[i].color * ks * k * pow(max(vec3::dot_product(R, S), 0.0f), 32);
        }
    }
    return ambient * trig.color + diffuse * trig.color + specular * trig.color;
}

__host__ __device__ uchar4 color_from_normalized(const vec3& v)
{
    float x = min(v.x, 1.0f);
    x = max(x, 0.0f);
    float y = min(v.y, 1.0f);
    y = max(y, 0.0f);
    float z = min(v.z, 1.0f);
    z = max(z, 0.0f);
    return make_uchar4(255. * x, 255. * y, 255. * z, 0u);
}

__host__ __device__ uchar4 ray(
    const vec3& pos, const vec3& dir,
    const Polygon* scene_trigs, int ntrigs,
    const Light* lights, int nlights)
{
    int k, k_min = -1;
    float ts_min;
    for (k = 0; k < ntrigs; k++) {
        vec3 e1 = scene_trigs[k].b - scene_trigs[k].a;
        vec3 e2 = scene_trigs[k].c - scene_trigs[k].a;
        vec3 p = vec3::cross_product(dir, e2);
        float div = vec3::dot_product(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = pos - scene_trigs[k].a;
        float u = vec3::dot_product(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = vec3::cross_product(t, e1);
        float v = vec3::dot_product(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        float ts = vec3::dot_product(q, e2) / div;
        if (ts < 0.0)
            continue;
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
    if (k_min == -1)
        return { 0, 0, 0, 0 };
    return color_from_normalized(phong_model(((dir * ts_min) + pos), dir,
        scene_trigs[k_min], scene_trigs,
        ntrigs, lights, nlights));
}

__global__ void render_cuda_kernel(
    uchar4* data, int w, int h,
    vec3 pc, vec3 pv, float angle,
    const Polygon* scene_trigs, int ntrigs,
    Light* lights, int nlights)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    float dw = 2.0 / (w - 1.0);
    float dh = 2.0 / (h - 1.0);
    float z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = (pv - pc).normalize();
    vec3 bx = vec3::cross_product(bz, { 0.0, 0.0, 1.0 }).normalize();
    vec3 by = vec3::cross_product(bx, bz).normalize();
    for (int j = id_y; j < h; j += offset_y)
        for (int i = id_x; i < w; i += offset_x) {
            vec3 v(-1.0f + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3 dir = vec3(
                bx.x * v.x + by.x * v.y + bz.x * v.z,
                bx.y * v.x + by.y * v.y + bz.y * v.z,
                bx.z * v.x + by.z * v.y + bz.z * v.z)
                           .normalize();
            data[(h - 1 - j) * w + i] = ray(pc, dir, scene_trigs, ntrigs, lights, nlights);
        }
}

CUDARenderer::CUDARenderer(Scene scene)
    : Renderer(scene)
{
    size_t poly_size = sizeof(Polygon) * poly.size();
    CUDA_ERR(cudaMalloc(&dev_poly, poly_size));
    CUDA_ERR(cudaMemcpy(dev_poly, poly.data(), poly_size, cudaMemcpyHostToDevice));

    debug("lights %ld", scene.lights.size());

    size_t lights_size = sizeof(Light) * scene.lights.size();
    CUDA_ERR(cudaMalloc(&dev_lights, lights_size));
    CUDA_ERR(cudaMemcpy(dev_lights, scene.lights.data(), lights_size, cudaMemcpyHostToDevice));

    CUDA_ERR(cudaMalloc(&dev_render, sizeof(uchar4) * render_size));
    CUDA_ERR(cudaMalloc(&dev_ssaa, sizeof(uchar4) * scene.w * scene.h));
}

CUDARenderer::~CUDARenderer()
{
#define cu_free(val) CUDA_ERR(cudaFree(val))
    cu_free(dev_poly);
    cu_free(dev_lights);
    cu_free(dev_render);
    cu_free(dev_ssaa);
#undef cu_free
}

#define NBLOCKSD2 dim3(16, 16)
#define NTHREADSD2 dim3(16, 16)

void CUDARenderer::Render(int frame)
{
    std::pair<vec3, vec3> p = cum_view(frame);
    START_KERNEL((render_cuda_kernel<<<NBLOCKSD2, NTHREADSD2>>>(
        dev_render, render_w, render_h,
        p.first, p.second, scene.angle,
        dev_poly, int(poly.size()),
        dev_lights, scene.lights.size())));
    CUDA_ERR(cudaDeviceSynchronize());
    START_KERNEL((ssaa_kernel<<<NBLOCKSD2, NTHREADSD2>>>(
        dev_ssaa, dev_render,
        scene.w, scene.h,
        render_w, render_h)));
    CUDA_ERR(cudaMemcpy(data_ssaa.data(), dev_ssaa,
        sizeof(uchar4) * scene.w * scene.h,
        cudaMemcpyDeviceToHost));
}