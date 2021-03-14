#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_ERR(k)                                               \
    do {                                                          \
        cudaError_t call = (k);                                   \
        if (call != cudaSuccess) {                                \
            fprintf(stderr, "CUDA ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(call));    \
            exit(0);                                              \
        }                                                         \
    } while (false)

#define KERNEL_ERR() CUDA_ERR(cudaGetLastError())

#endif
