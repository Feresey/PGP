#include <cmath>

#include <thrust/detail/raw_pointer_cast.h>

#include "helpers.cuh"

#define WARP_SIZE 32
#define SIDE_SIZE 32

__global__ void transponse_kernel(
    double* out, const double* A,
    const uint32_t n, const uint32_t m)
{
    __shared__ double shared[WARP_SIZE][WARP_SIZE + 1];
    uint x = blockIdx.x * WARP_SIZE + threadIdx.x,
         y = blockIdx.y * WARP_SIZE + threadIdx.y;

#pragma unroll
    for (int k = 0; k < WARP_SIZE; k += SIDE_SIZE) {
        if (x < n && y + k < m) {
            shared[threadIdx.y + k][threadIdx.x] = A[((y + k) * n) + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * WARP_SIZE + threadIdx.x;
    y = blockIdx.x * WARP_SIZE + threadIdx.y;

#pragma unroll
    for (int k = 0; k < WARP_SIZE; k += SIDE_SIZE) {
        if (x < m && y < n) {
            out[((y + k) * m) + x] = shared[threadIdx.x][threadIdx.y + k];
        }
    }

    // const uint tid_x = threadIdx.x,
    //            tid_y = threadIdx.y;
    // const uint bx = blockIdx.x * WARP_SIZE,
    //            by = blockIdx.y * WARP_SIZE;
    // const uint i = blockDim.x * blockIdx.x + threadIdx.x,
    //            j = blockDim.y * blockIdx.y + threadIdx.y;
    // const uint idy_T = blockDim.x * blockIdx.x + threadIdx.y,
    //            idx_T = blockDim.y * blockIdx.y + threadIdx.x;

    // // temp
    // // const uint bx = blockIdx.x, by = blockIdx.y;

    // __shared__ double shared[WARP_SIZE][WARP_SIZE + 1];

    // if (idx < n && idy < m) {
    //     shared[tid_x][tid_y] = A[idx * m + idy];
    //     // printf("%2d %2d in : %02d %02d %lf\n", bx, by, tid_x, tid_y, shared[tid_x][tid_y]);
    //     //     printf("%d\tidx  : %d\tidy  : %d\t%lf\n", block, idx, idy, shared[tid_x][tid_y]);
    //     // } else {
    //     //     printf("%d\tidx  : %d\tidy  : %d\tfailed\n", block, idx, idy);
    // }

    // __syncthreads();
    // if (idx_T < m && idy_T < n) {
    //     out[idx_T * n + idy_T] = shared[tid_y][tid_x];
    //     // printf("%2d %2d out: %02d %02d %lf\n", bx, by, tid_y, tid_x, shared[tid_y][tid_x]);
    //     //     printf("%d\tidx_t: %d\tidy_t: %d\t%lf\n", block, idx_T, idy_T, shared[tid_y][tid_x]);
    //     // } else {
    //     //     printf("%d\tidx_t: %d\tidy_t: %d\tfailed\n", block, idx_T, idy_T);
    // }
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