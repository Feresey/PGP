#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.h"

#define EPS 1e-3
#define IN_EPS(EQ) (abs(EQ) < EPS)

// да, костыль
int blocks = 1;
int threads = 32;

__device__ __constant__ float4 dev_centers[500];

__device__ float distance(const uchar4& a, const float4& b)
{
    // printf("%f %f %f\n", b.x, b.y, b.z);
    float x = b.x - a.x,
          y = b.y - a.y,
          z = b.z - a.z;
    return x * x + y * y + z * z;
}

__device__ int calc_best_distance(const uchar4& point, const int n_classes)
{
    int best_class = 255;
    float best_distance = 690420;

    for (int i = 0; i < n_classes; i++) {
        float curr_distance = distance(point, dev_centers[i]);
        // printf("compare %f<>%f\n", curr_distance, best_distance);
        if (curr_distance < best_distance) {
            best_class = i;
            best_distance = curr_distance;
        }
    }

    return best_class;
}

__device__ float norm(float4 a, float4 b)
{
    float x = a.x - b.x,
          y = a.y - b.y,
          z = a.z - b.z;
    return x * x + y * y + z * z;
}

// классификация пикселей по текущим центрам групп.
__global__ void kernel(
    uchar4* data, int n,
    float4* new_centers, ulonglong4* cache, int n_classes,
    unsigned long long* equal)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = id; i < n; i += offset) {
        data[i].w = calc_best_distance(data[i], n_classes);

        // вычисление новых центров классов
        // суммирование значений пикселей по классам
        uchar4 elem = data[i];
        ulonglong4* cache_elem = &cache[elem.w];
        atomicAdd(&cache_elem->x, elem.x);
        atomicAdd(&cache_elem->y, elem.y);
        atomicAdd(&cache_elem->z, elem.z);
        atomicAdd(&cache_elem->w, 1);
    }

    __syncthreads();
    // присваивание новых значений центров классов.
    for (int i = id; i < n_classes; i += offset) {
        ulonglong4 cache_elem = cache[i];
        float4 elem = make_float4(
            float(cache_elem.x) / cache_elem.w,
            float(cache_elem.y) / cache_elem.w,
            float(cache_elem.z) / cache_elem.w,
            0.0f);
        new_centers[i] = elem;

        // условие сходимости -- центры не изменились
        float4 old = dev_centers[i];
        if (norm(old, elem) > EPS) {
            // printf("%f %f %f <> %f %f %f\n", old.x, old.y, old.z, elem.x, elem.y, elem.z);
            atomicAdd(equal, 1);
        }
    }
}

typedef struct {
    int x, y;
} Center;

void launch_k_means(uchar4* host_data, const int w, const int h, const Center* start_centers, const int n_classes)
{
    const int n = h * w;

    float4* dev_next_centers;
    uchar4* dev_data;
    ulonglong4* dev_cache;

    CSC(cudaMalloc(&dev_next_centers, sizeof(float4) * n));
    CSC(cudaMalloc(&dev_cache, sizeof(ulonglong4) * n));

    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * n));
    CSC(cudaMemcpy(dev_data, host_data, sizeof(uchar4) * n, cudaMemcpyHostToDevice));

    {
        float4* tmp_centers;
        // значения указанных пикселей.
        tmp_centers = (float4*)malloc(sizeof(uchar4) * n);
        for (int i = 0; i < n_classes; i++) {
            uchar4 elem = host_data[start_centers[i].y * w + start_centers[i].x];
            tmp_centers[i] = make_float4(elem.x, elem.y, elem.z, 0.0f);
        }
        // printf("init\n");
        // for (int i = 0; i < n_classes; i++) {
        //     uchar4 m = tmp_centers[i];
        // printf("%d %d %d\n", m.x, m.y, m.z);
        // }
        // printf("\n\n");
        CSC(cudaMemcpy(dev_next_centers, tmp_centers, sizeof(float4) * n, cudaMemcpyHostToDevice));
        free(tmp_centers);
    }

    unsigned long long equal = 0,
                       *dev_equal;
    CSC(cudaMalloc(&dev_equal, sizeof(unsigned long long)));

    int killme = 5;
    while (killme--) {
        CSC(cudaMemcpyToSymbol(dev_centers, dev_next_centers, sizeof(float4) * n_classes, 0, cudaMemcpyDeviceToDevice));

        // CSC(cudaMemcpy(host_data, dev_data, sizeof(uchar4) * n, cudaMemcpyDeviceToHost));
        // printf("===\n");
        // for (int x = 0; x < w; x++) {
        //     for (int y = 0; y < h; y++) {
        //         printf("%d ", x * w + y);
        //         uchar4 p = host_data[x * w + y];
        //         printf("%02x%02x%02x %d ", p.x, p.y, p.z, p.w);
        //     }
        //     printf("\n");
        // }
        // printf("===\n");
        // for (int i = 0; i < n; i++) {
        //     printf("%d ", i);
        //     uchar4 p = host_data[i];
        //     printf("%d %d %d %d\n", p.x, p.y, p.z, p.w);
        // }
        // printf("===\n");

        CSC(cudaMemset(dev_equal, 0, sizeof(unsigned long long)));
        CSC(cudaMemset(dev_cache, 0, sizeof(ulonglong4) * n_classes));

        START_KERNEL((kernel<<<blocks, threads>>>(
            dev_data, n,
            dev_next_centers, dev_cache, n_classes,
            dev_equal)));

        CSC(cudaMemcpy(&equal, dev_equal, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        // printf("equal: %llu\n", equal);
        if (equal == 0) {
            break;
        }
    }

    CSC(cudaFree(dev_equal));
    CSC(cudaMemcpy(host_data, dev_data, sizeof(uchar4) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_next_centers));
    CSC(cudaFree(dev_cache));
}

int main()
{
#ifdef BENCHMARK
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-blocks") == 0) {
            blocks = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-threads") == 0) {
            threads = atoi(argv[i + 1]);
        }
    }
#endif

    char input[100], output[100];

    scanf("%s", input);
    scanf("%s", output);

    int n_classes;
    Center* centers;
    scanf("%d", &n_classes);
    centers = (Center*)malloc(n_classes * sizeof(Center));
    for (int i = 0; i < n_classes; i++) {
        scanf("%d", &centers[i].x);
        scanf("%d", &centers[i].y);
    }

    FILE* in = fopen(input, "rb");
    if (ferror(in)) {
        printf("ERROR opening input file: %s\n", input);
        exit(0);
    }

    uchar4* data;
    uint32_t w, h;
    uint32_t err;
    err = read_image(in, &data, &w, &h);
    if (err != 0) {
        printf("ERROR in %s:%d scan image: %d", __FILE__, __LINE__, err);
        exit(0);
    }
    fclose(in);

    launch_k_means(data, h, w, centers, n_classes);

    FILE* out = fopen(output, "wb");
    err = write_image(out, data, w, h);
    if (err != 0) {
        printf("ERROR in %s:%d write image: %d", __FILE__, __LINE__, err);
    }
    fclose(out);

    free(data);
    return 0;
}