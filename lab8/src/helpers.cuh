#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <cstdio>

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

#define debug(format, args...)                                              \
    do {                                                                    \
        fprintf(stderr, "%s:%d\t" format "\n", __FILE__, __LINE__, ##args); \
        fflush(stderr);                                                     \
    } while (false)

#ifndef __NVCC__
#include "dim3/dim3.hpp"

#undef START_KERNEL
#define START_KERNEL(KERNEL)

#define __host__
#define __device__
#define __global__
#define __shared__

typedef int cudaError_t;

#define cudaSuccess 0

typedef mydim3<int> dim3;

static const dim3 threadIdx, threadDim, blockIdx, blockDim, gridIdx, gridDim;

cudaError_t cudaGetLastError();
char* cudaGetErrorString(cudaError_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceCount(int*);

void __syncthreads();

#define cudaMemcpyDeviceToHost 0
#define cudaMemcpyHostToDevice 1

cudaError_t cudaMemcpy(void*, void*, int, int);
template <class abuse>
cudaError_t cudaMalloc(abuse, int);
cudaError_t cudaFree(void*);
#endif

#endif // HELPERS_CUH
