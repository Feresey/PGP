#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define CHECK(SIZE_EQ, WANT)    \
    {                           \
        size_t err = (SIZE_EQ); \
        if (err != WANT) {      \
            return err;         \
        };                      \
    }

int read_image(FILE* f, uint32_t** out, uint32_t* w, uint32_t* h)
{
    CHECK(fread(w, sizeof(uint32_t), 1, f), 1);
    CHECK(fread(h, sizeof(uint32_t), 1, f), 1);
    const size_t size = *w * *h;
    *out = (uint32_t*)malloc(size * sizeof(uint32_t));
    CHECK(fread(*out, sizeof(uint32_t), size, f), size);
    return 0;
}

int write_image(FILE* f, uint32_t* data, uint32_t w, uint32_t h)
{
    CHECK(fwrite(&w, sizeof(uint32_t), 1, f), 1);
    CHECK(fwrite(&h, sizeof(uint32_t), 1, f), 1);
    CHECK(fwrite(data, sizeof(uint32_t), w * h, f), w * h);
    return 0;
}

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

__device__ float norm(uchar4 u)
{
    return roundf(0.299*float(u.x) + 0.587*float(u.y) + 0.144*float(u.z));
}

__global__ void kernel(uchar4* out, uint32_t w, uint32_t h)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    float z[9];
    int left, right, top, bottom;
    float g_x, g_y;
    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {

            left = x == 0 ? 0 : x - 1;
            right = x == (h - 1) ? (h - 1) : x + 1;
            top = y == 0 ? 0 : y - 1;
            bottom = y == (w - 1) ? (w - 1) : y + 1;

            z[0] = norm(tex2D(tex, left, top));
            z[1] = norm(tex2D(tex, x, top));
            z[2] = norm(tex2D(tex, right, top));

            z[3] = norm(tex2D(tex, left, y));
            z[4] = norm(tex2D(tex, x, y));
            z[5] = norm(tex2D(tex, right, y));

            z[6] = norm(tex2D(tex, left, bottom));
            z[7] = norm(tex2D(tex, x, bottom));
            z[8] = norm(tex2D(tex, right, bottom));

            g_x = (z[6] + z[7] + z[8]) - (z[0] + z[1] + z[2]);
            g_y = (z[2] + z[5] + z[8]) - (z[0] + z[3] + z[6]);

            unsigned char res = (unsigned char)(__fsqrt_ru((g_x * g_x) + (g_y * g_y)));
            out[x + y * w] = make_uchar4(res, res, res, 255);
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
    if (ferror(in)) {
        printf("ERROR opening input file: %s\n", input);
        exit(0);
    }

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
