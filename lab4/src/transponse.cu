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

    // temp
    // const uint block = blockIdx.x * blockDim.x + blockIdx.y;

    __shared__ double shared[32][32 + 1];

    if (idx < n && idy < m) {
        shared[tid_x][tid_y] = A[idx * m + idy];
        //     printf("%d\tidx  : %d\tidy  : %d\t%lf\n", block, idx, idy, shared[tid_x][tid_y]);
        // } else {
        //     printf("%d\tidx  : %d\tidy  : %d\tfailed\n", block, idx, idy);
    }

    __syncthreads();
    if (idy_T < m && idx_T < n) {
        out[idy_T * n + idx_T] = shared[tid_y][tid_x];
        //     printf("%d\tidx_t: %d\tidy_t: %d\t%lf\n", block, idx_T, idy_T, shared[tid_y][tid_x]);
        // } else {
        //     printf("%d\tidx_t: %d\tidy_t: %d\tfailed\n", block, idx_T, idy_T);
    }
}

template <class T>
T div_up(T a, T b) { return (a - 1) / b + 1; }

dev_matrix transponse(const dev_matrix& A, const uint32_t n, const uint32_t m)
{
    const double* raw = thrust::raw_pointer_cast(&A[0]);
    dev_matrix res(m * n);
    double* res_raw = thrust::raw_pointer_cast(&res[0]);
    const dim3 blocks = dim3(div_up<uint>(n, 32), div_up<uint>(m, 32));
    const dim3 threads = dim3(32, 32);

    // show_matrix(stderr, A, n, m);
    fprintf(stderr, "n: %d, m: %d\n", n, m);
    START_KERNEL((transponse_kernel<<<blocks, threads>>>(res_raw, raw, n, m)));

    return res;
}