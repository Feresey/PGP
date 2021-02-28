#include <cmath>

#include <thrust/detail/raw_pointer_cast.h>

#include "helpers.cuh"

__global__ void transponse_kernel(
    double* out, const double* A,
    const uint32_t n, const uint32_t m)
{
    const uint tid_x = threadIdx.x,
               tid_y = threadIdx.y;
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x,
               idy = blockDim.y * blockIdx.y + threadIdx.y;
    const uint idx_T = blockDim.x * blockIdx.x + threadIdx.y,
               idy_T = blockDim.y * blockIdx.y + threadIdx.x;
    const bool allow = idx < n && idy < m;

    __shared__ double shared[32][32 + 1];

    // if (tid_x == 0 && tid_y == 2) {
    //     printf("n: %d, m: %d\n", n, m);
    //     printf("(%d %d):(%d %d)\n", tid_x, tid_y, idx, idy);
    // }

    if (allow) {
        // printf("%d %d: %f\n", idx, idy, A[idy * m + idx]);
        shared[tid_x][tid_y] = A[idy * m + idx];
    }

    __syncthreads();
    if (allow) {
        out[idx_T * n + idy_T] = shared[tid_y][tid_x];
    }
}

template <class T>
T div_up(T a, T b) { return (a - 1) / b + 1; }

dev_matrix transponse(const dev_matrix& A, const uint32_t n, const uint32_t m)
{
    const double* A_raw = thrust::raw_pointer_cast(&A[0]);
    dev_matrix A_trans(m * n);
    double* A_trans_raw = thrust::raw_pointer_cast(&A_trans[0]);
    const dim3 blocks = dim3(div_up<uint>(n, 32), div_up<uint>(m, 32));
    const dim3 threads = dim3(32, 32);

    START_KERNEL((transponse_kernel<<<blocks, threads>>>(A_trans_raw, A_raw, n, m)));

    return A_trans;
}