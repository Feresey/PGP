#ifndef HELPERS_H
#define HELPERS_H

#define CSC(call)                                              \
    do {                                                       \
        if (call != cudaSuccess) {                             \
            fprintf(stderr,                                    \
                "ERROR in %s:%d. Message: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(call)); \
            exit(0);                                           \
        }                                                      \
    } while (0)

#ifndef BENCHMARK
#define START_KERNEL(BLOCKS, THREADS, KERNEL, ...) KERNEL<<<BLOCKS, THREADS>>>(__VA_ARGS__);
#else
#define START_KERNEL(BLOCKS, THREADS, KERNEL, ...) \
cudaEvent_t start, end; \
CSC(cudaEventCreate(&start)); \
CSC(cudaEventCreate(&end)); \
CSC(cudaEventRecord(start)); \
fprintf(stderr, "blocks = %d\nthreads = %d\n", blocks, threads); \
KERNEL<<<BLOCKS, THREADS>>>(__VA_ARGS__); \
CSC(cudaGetLastError()); \
CSC(cudaEventRecord(end)); \
CSC(cudaEventSynchronize(end)); \
float t; \
CSC(cudaEventElapsedTime(&t, start, end)); \
CSC(cudaEventDestroy(start)); \
CSC(cudaEventDestroy(end)); \
fprintf(stderr, "time = %010.6f\n", t);
#endif // BENCHMARK

#endif // HELPERS_H