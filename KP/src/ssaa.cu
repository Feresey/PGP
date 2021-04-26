#include "ssaa.hpp"

#include "vec/vec3.hpp"

__host__ __device__ uchar4 ssaa_algo(
    uchar4* data,
    int i, int j,
    int w, int h,
    int kernel_w, int kernel_h)
{
    vec3 res;
    for (int y = i; y < i + kernel_h; ++y) {
        for (int x = j; x < j + kernel_w; ++x) {
            auto pix = data[y * w + x];
            res = res + vec3(pix.x, pix.y, pix.z);
        }
    }
    auto pix = res * (1.0 / (kernel_w * kernel_h));
    return make_uchar4(pix.x, pix.y, pix.z, 255);
}

__global__ void ssaa_kernel(
    uchar4* dst, uchar4* src,
    int new_w, int new_h,
    int w, int h)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    int kernel_w = w / new_w;
    int kernel_h = h / new_h;

    for (int i = id_y; i < new_h; i += offset_y) {
        for (int j = id_x; j < new_w; j += offset_x) {
            int pix_i = i * kernel_h;
            int pix_j = j * kernel_w;

            dst[i * new_w + j] = ssaa_algo(src, pix_i, pix_j, w, h, kernel_w, kernel_h);
        }
    }
}

void ssaa_omp(
    uchar4* dst, uchar4* src,
    int new_w, int new_h,
    int w, int h)
{
    int kernel_w = w / new_w, kernel_h = h / new_h;
#pragma omp parallel for
    for (int pix = 0; pix < new_w * new_h; ++pix) {
        int i = pix / new_w;
        int j = pix % new_w;

        int pix_i = i * kernel_h;
        int pix_j = j * kernel_w;

        dst[i * new_w + j] = ssaa_algo(src, pix_i, pix_j, w, h, kernel_w, kernel_h);
    }
}
