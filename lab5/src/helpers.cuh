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
#define START_KERNEL(KERNEL)                       \
    do {                                           \
        cudaEvent_t start, end;                    \
        CSC(cudaEventCreate(&start));              \
        CSC(cudaEventCreate(&end));                \
        CSC(cudaEventRecord(start));               \
        KERNEL;                                    \
        CSC(cudaGetLastError());                   \
        CSC(cudaEventRecord(end));                 \
        CSC(cudaEventSynchronize(end));            \
        float t;                                   \
        CSC(cudaEventElapsedTime(&t, start, end)); \
        CSC(cudaEventDestroy(start));              \
        CSC(cudaEventDestroy(end));                \
        fprintf(stderr, "time = %010.6f\n", t);    \
    } while (false)
#endif // BENCHMARK

#define SWAP(arr, a, b)    \
    do {                   \
        int temp = arr[a]; \
        arr[a] = arr[b];   \
        arr[b] = temp;     \
    } while (false)

#ifdef __cplusplus
extern "C" {
#endif

uint32_t scan_4();
void print_arr(FILE* out, const int* arr, const uint32_t size);

#ifdef __cplusplus
};
#endif

#endif // HELPERS_H