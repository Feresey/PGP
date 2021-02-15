#ifndef HELPERS_H
#define HELPERS_H

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

int read_image(FILE* f, uchar4** out, uint32_t* w, uint32_t* h)
{
    fread(w, (size_t)sizeof(uint32_t), 1UL, f);
    fread(h, (size_t)sizeof(uint32_t), 1UL, f);
    const size_t size = *w * *h;
    *out = (uchar4*)malloc(size * 4);
    fread(*out, (size_t)sizeof(uchar4), (size_t)size, f);
    return 0;
}

int write_image(FILE* f, uchar4* data, uint32_t w, uint32_t h)
{
    fwrite(&w, (size_t)sizeof(uint32_t), 1UL, f);
    fwrite(&h, (size_t)sizeof(uint32_t), 1UL, f);
    fwrite(data, sizeof(uchar4), w * h, f);
    return 0;
}

#endif // HELPERS_H