#ifndef HELPERS_H
#define HELPERS_H

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

#define CHECK(SIZE_EQ, WANT)    \
    {                           \
        size_t err = (SIZE_EQ); \
        if (err != WANT) {      \
            return err;         \
        };                      \
    }

int read_image(FILE* f, uchar4** out, uint32_t* w, uint32_t* h)
{
    CHECK(fread(w, sizeof(uint32_t), 1, f), 1);
    CHECK(fread(h, sizeof(uint32_t), 1, f), 1);
    const size_t size = *w * *h;
    *out = (uchar4*)malloc(size * 4);
    CHECK(fread(*out, sizeof(uchar4), size, f), size);
    return 0;
}

int write_image(FILE* f, uchar4* data, uint32_t w, uint32_t h)
{
    CHECK(fwrite(&w, sizeof(uint32_t), 1, f), 1);
    CHECK(fwrite(&h, sizeof(uint32_t), 1, f), 1);
    CHECK(fwrite(data, sizeof(uchar4), w * h, f), w * h);
    return 0;
}

#endif // HELPERS_H