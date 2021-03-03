#include <cmath>

#include <thrust/detail/raw_pointer_cast.h>

#include "helpers.cuh"

#define WARP_SIZE 32

__global__ void transponse_kernel(
    double* out, const double* A,
    const uint32_t n, const uint32_t m)
{
    uint i = blockIdx.x * WARP_SIZE + threadIdx.x,
         j = blockIdx.y * WARP_SIZE + threadIdx.y;

    __shared__ double shared[WARP_SIZE][WARP_SIZE + 1];

    if (i < n && j < m) {
        shared[threadIdx.x][threadIdx.y] = A[i * m + j];
    }

    i = blockIdx.y * WARP_SIZE + threadIdx.x;
    j = blockIdx.x * WARP_SIZE + threadIdx.y;
    __syncthreads();
    if (i < m && j < n) {
        out[i * n + j] = shared[threadIdx.y][threadIdx.x];
    }
}

template <class T>
T div_up(T a, T b) { return (a - 1) / b + 1; }

dev_matrix transponse(const dev_matrix& A, const uint32_t n, const uint32_t m)
{
    const double* raw = thrust::raw_pointer_cast(&A[0]);
    dev_matrix res(m * n);
    double* res_raw = thrust::raw_pointer_cast(&res[0]);
    const dim3 blocks = dim3(div_up<uint>(n, WARP_SIZE), div_up<uint>(m, WARP_SIZE));
    const dim3 threads = dim3(WARP_SIZE, WARP_SIZE);

    START_KERNEL((transponse_kernel<<<blocks, threads, (WARP_SIZE) * (WARP_SIZE + 1)>>>(res_raw, raw, n, m)));

    return res;
}