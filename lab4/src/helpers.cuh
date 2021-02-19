#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <stdint.h>
#include <stdio.h>

#define CSC(k)                                                 \
    do {                                                       \
        cudaError_t call = k;                                  \
        if (call != cudaSuccess) {                             \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n",   \
                __FILE__, __LINE__, cudaGetErrorString(call)); \
            exit(0);                                           \
        }                                                      \
    } while (false)

#ifndef BENCHMARK
#define START_KERNEL(KERNEL) \
    KERNEL;                  \
    CSC(cudaGetLastError())
#else
#define START_KERNEL(KERNEL)                                         \
    cudaEvent_t start, end;                                          \
    CSC(cudaEventCreate(&start));                                    \
    CSC(cudaEventCreate(&end));                                      \
    CSC(cudaEventRecord(start));                                     \
    fprintf(stderr, "blocks = %d\nthreads = %d\n", blocks, threads); \
    KERNEL;                                                          \
    CSC(cudaGetLastError());                                         \
    CSC(cudaEventRecord(end));                                       \
    CSC(cudaEventSynchronize(end));                                  \
    float t;                                                         \
    CSC(cudaEventElapsedTime(&t, start, end));                       \
    CSC(cudaEventDestroy(start));                                    \
    CSC(cudaEventDestroy(end));                                      \
    fprintf(stderr, "time = %010.6f\n", t)
#endif // BENCHMARK

#ifdef __NVCC__
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef thrust::host_vector<double> host_matrix;
typedef thrust::device_vector<double> dev_matrix;

#define BLOCKS dim3(8, 8)
#define THREADS dim3(8, 8)
#else
#include "helpers.hpp"
#endif

void read_matrix(host_matrix& out, const size_t n, const size_t m);
void show_matrix(FILE* out, const host_matrix& data, const size_t n, const size_t m);

dev_matrix inverse(const dev_matrix& matrix, const size_t n, const size_t m);
dev_matrix solve(const dev_matrix& A, const dev_matrix& B, const size_t n, const size_t m, const size_t k);

#endif // HELPERS_H