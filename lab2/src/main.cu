#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "serialize.h"

#define CSC(kal)                                               \
    do {                                                       \
        auto call = kal;                                       \
        if (call != cudaSuccess) {                             \
            fprintf(stderr,                                    \
                "ERROR in %s:%d. Message: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(call)); \
            exit(0);                                           \
        }                                                      \
    } while (0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ int4 uchar4sum(uchar4 a, uchar4 b, uchar4 c)
{
    int4 res;
    res.x = int(a.x) + int(b.x) + int(c.x);
    res.y = int(a.y) + int(b.y) + int(c.y);
    res.z = int(a.z) + int(b.z) + int(c.z);
    res.w = 0;
    return res;
}

__device__ int4 int4sub(int4 a, int4 b)
{
    int4 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    res.z = a.z - b.z;
    res.w = 0;
    return res;
}

#define norm(u) (0.299 * float(u.x) + 0.587 * float(u.y) + 0.114 * float(u.z))
#define meanless(a, b) sqrtf(float(a * a) + float(b * b))

__device__ float prewitt(uchar4* z)
{
    int4 up = uchar4sum(z[0], z[1], z[2]);
    int4 down = uchar4sum(z[6], z[7], z[8]);

    int4 right = uchar4sum(z[0], z[3], z[6]);
    int4 left = uchar4sum(z[2], z[5], z[8]);

    int4 g_x = int4sub(down, up);
    int4 g_y = int4sub(left, right);

    return meanless(norm(g_x), norm(g_y));
}

__global__ void kernel(uchar4* out, uint32_t w, uint32_t h)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    uchar4 z[9];
    int left, right, top, bottom;
    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {

            left = x - 1;
            right = x + 1;
            top = y - 1;
            bottom = y + 1;

            z[0] = tex2D(tex, left, top);
            z[1] = tex2D(tex, x, top);
            z[2] = tex2D(tex, right, top);
            z[3] = tex2D(tex, left, y);
            z[4] = tex2D(tex, x, y);
            z[5] = tex2D(tex, right, y);
            z[6] = tex2D(tex, left, bottom);
            z[7] = tex2D(tex, x, bottom);
            z[8] = tex2D(tex, right, bottom);

            float res = prewitt(z);
            unsigned char res_byte = res;
            if (res < 0) {
                printf("ERROR: ты обосрался: %f\n", res);
            }
            if (res > 255) {
                res_byte = 255;
            }
            out[x + y * w] = make_uchar4(res_byte, res_byte, res_byte, 0);
        }
    }
}

#ifdef BENCHMARK
int main(int argc, char** argv)
#else
int main()
#endif
{
    unsigned int blocks = 1;
    unsigned int threads = 8;
#ifdef BENCHMARK
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-blocks") == 0) {
            blocks = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-threads") == 0) {
            threads = atoi(argv[i + 1]);
        }
    }
#endif

    char input[255], output[255];

    scanf("%s", input);
    scanf("%s", output);

    FILE* in = fopen(input, "rb");
    if (in == NULL || ferror(in)) {
        perror(NULL);
        printf("ERROR opening input file: %s\n", input);
        exit(0);
    }

    uint32_t* data;
    uint32_t w, h;
    int err;
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
