#include <cmath>

#include <thrust/detail/raw_pointer_cast.h>

#include "helpers.cuh"

#define WARP_SIZE 32
#define HALF_WARP_SIZE 16

__global__ void transponse_kernel(
    double* out, const double* A,
    const uint32_t n, const uint32_t m)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x,
         j = blockIdx.y * blockDim.y + threadIdx.y;

    uint idx = threadIdx.x,
         idy = threadIdx.y;

    // При записи кусочка матрицы в разделяемую панять конфликта банков не будет.
    // Зато при чтении колонки будет конфликт 32 порядка - ведь все элементы в 1 банке.

    // выравнивание банков.
    // так выглядит распределение банков по памяти
    /*
        0  1  2  3 ... 31 |  0
        1  2  3  4 ...  0 |  1
        2  3  4  5 ...  1 |  2
        3  4  5  6 ...  2 |  3
        ...        ...    |   
        31 0  1  2 ... 30 | 31
    */
    // В таком случае последняя колонка не будет использоваться.
    // Но зато при чтении колонок элементы будут принадлежать разным банкам и конфликта не будет.

    __shared__ double shared[WARP_SIZE + 1][WARP_SIZE + 1];

    if (i < n && j < m) {
        if (idx < 16 && idy < 16)
            shared[idx][idy] = A[i * m + j];
        else if (idx < 16 && idy >= 16)
            shared[idx][idy + 1] = A[i * m + j];
        else if (idx >= 16 && idy < 16)
            shared[idx + 1][idy] = A[i * m + j];
        else
            shared[idx + 1][idy + 1] = A[i * m + j];
    }

    // изменение индексации, чтобы запись прошла транзакцией
    i = blockIdx.y * blockDim.y + threadIdx.x;
    j = blockIdx.x * blockDim.x + threadIdx.y;
    __syncthreads();
    if (i < m && j < n) {
        if (idx < 16 && idy < 16)
            out[i * n + j] = shared[idy][idx];
        else if (idx < 16 && idy >= 16)
            out[i * n + j] = shared[idy + 1][idx];
        else if (idx >= 16 && idy < 16)
            out[i * n + j] = shared[idy][idx + 1];
        else
            out[i * n + j] = shared[idy + 1][idx + 1];
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

    START_KERNEL((transponse_kernel<<<blocks, threads>>>(res_raw, raw, n, m)));

    return res;
}