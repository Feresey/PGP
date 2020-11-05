#include <stdio.h>

#define CSC(call)                                                                                           \
    do {                                                                                                    \
        if (call != cudaSuccess) {                                                                          \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); \
            exit(0);                                                                                        \
        }                                                                                                   \
    } while (0)

__global__ void inverse(float* out, float* vec, int n)
{
    int i, idx = blockDim.x * blockIdx.x + threadIdx.x; // Абсолютный номер потока
    int offset = blockDim.x * gridDim.x; // Общее кол-во потоков

    for (i = idx; i < n; i += offset) {
        out[n - i - 1] = vec[i];
    }
}

int main(int argc, char** argv)
{
    int blocks = 1;
    int threads = 32;
#ifdef BENCHMARK
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-blocks") == 0) {
            blocks = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-threads") == 0) {
            threads = atoi(argv[i + 1]);
        }
    }
#endif

    int n;
    scanf("%d", &n);
    float* vec = (float*)(malloc(sizeof(float) * n));

    for (int i = 0; i < n; i++) {
        scanf("%f", &vec[i]);
    }

    float *dev_vec, *dev_out;
    CSC(cudaMalloc(&dev_vec, sizeof(float) * n));
    CSC(cudaMalloc(&dev_out, sizeof(float) * n));
    CSC(cudaMemcpy(dev_vec, vec, sizeof(float) * n, cudaMemcpyHostToDevice));

#ifdef BENCHMARK
    cudaEvent_t start, end;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&end));
    CSC(cudaEventRecord(start));

    fprintf(stderr, "blocks = %d\nthreads = %d\n", blocks, threads);
#endif

    inverse<<<blocks, threads>>>(dev_out, dev_vec, n);
    CSC(cudaGetLastError());

#ifdef BENCHMARK
    CSC(cudaEventRecord(end));
    CSC(cudaEventSynchronize(end));
    float t;
    CSC(cudaEventElapsedTime(&t, start, end));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(end));

    fprintf(stderr, "time = %010.6f\n", t);
#endif

    float* out = (float*)(malloc(sizeof(float) * n));
    CSC(cudaMemcpy(out, dev_out, sizeof(float) * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%.10e ", out[i]);
    }
    printf("\n");

    CSC(cudaFree(dev_vec));
    CSC(cudaFree(dev_out));
    free(vec);
    free(out);
}
