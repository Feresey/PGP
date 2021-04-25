#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <cstdio>
#include <ctime>

#define HD __host__ __device__

#define CUDA_ERR(k)                                               \
    do {                                                          \
        cudaError_t call = (k);                                   \
        if (call != cudaSuccess) {                                \
            fprintf(stderr, "CUDA ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(call));    \
            exit(0);                                              \
        }                                                         \
    } while (false)

#define START_KERNEL(KERNEL) \
    KERNEL;                  \
    CUDA_ERR(cudaGetLastError())

static const clock_t start_time = clock();

#define debug(format, args...)                                                                                                       \
    do {                                                                                                                             \
        clock_t curr_time = clock();                                                                                                 \
        fprintf(stderr, "%9.3f %s:%d\t" format "\n", (double)(curr_time - start_time) / CLOCKS_PER_SEC, __FILE__, __LINE__, ##args); \
        fflush(stderr);                                                                                                              \
    } while (false)

void setDevice(int);
void getDeviceCount(int*);

#ifndef __NVCC__
#include "dim3/dim3.hpp"

#undef START_KERNEL
#define START_KERNEL(KERNEL)

#define __host__
#define __device__
#define __global__
#define __shared__
#define __attribute__(...)

typedef int cudaError_t;

#define cudaSuccess 0

typedef mydim3<int> dim3;

struct uchar4 {
    u_char x, y, z, w;
};
uchar4 make_uchar4(u_char x, u_char y, u_char z, u_char w);

static const dim3 threadIdx, threadDim, blockIdx, blockDim, gridIdx, gridDim;

cudaError_t cudaGetLastError();
char* cudaGetErrorString(cudaError_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaDeviceSynchronize();

void __syncthreads();

#define cudaMemcpyDeviceToHost 0
#define cudaMemcpyHostToDevice 1

cudaError_t cudaMemcpy(void*, void*, int, int);
template <class abuse>
cudaError_t cudaMalloc(abuse, int);
cudaError_t cudaFree(void*);
#endif

#endif // HELPERS_CUH
