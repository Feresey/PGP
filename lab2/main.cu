#include <stdio.h>

#include "serialize.h"

#define CSC(call)                                              \
    do {                                                       \
        if (call != cudaSuccess) {                             \
            fprintf(stderr,                                    \
                "ERROR in %s:%d. Message: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(call)); \
            exit(0);                                           \
        }                                                      \
    } while (0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4* out, uint32_t w, uint32_t h)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {
            out[x + y * w] = uchar4(tex2D(tex, x, y));
        }
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

    char input[100], output[200];

    scanf("%s", input);
    scanf("%s", output);

    FILE* in = fopen(input, "rb");

    uint32_t* data;
    uint32_t w, h;
    uint32_t err;
    err = read_image(in, &data, &w, &h);
    if (err != 0) {
        printf("ERROR in %s:%d scan image: %d", __FILE__, __LINE__, err);
        exit(0);
    }
    fclose(in);

    cudaArray* arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;

    CSC(cudaBindTextureToArray(tex, arr, ch));
    uchar4* dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * h * w));

#ifdef BENCHMARK
    cudaEvent_t start, end;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&end));
    CSC(cudaEventRecord(start));

    fprintf(stderr, "blocks = %d\nthreads = %d\n", blocks, threads);
#endif

    kernel<<<dim3(blocks, blocks), dim3(threads, threads)>>>(dev_data, w, h);
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

    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));

    FILE* out = fopen(output, "wb");
    err = write_image(out, data, w, h);
    if (err != 0) {
        printf("ERROR in %s:%d write image: %d", __FILE__, __LINE__, err);
    }
    fclose(out);

    CSC(cudaUnbindTexture(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_data));
    free(data);
}
