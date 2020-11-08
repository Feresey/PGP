#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.h"

#define CHECK(SIZE_EQ, WANT)    \
    {                           \
        size_t err = (SIZE_EQ); \
        if (err != WANT) {      \
            return err;         \
        };                      \
    }

// да, костыль
int blocks = 1;
int threads = 32;

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

__device__ float distance(const uchar4 a, const uchar4 b)
{
    uchar4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    return sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
}

__device__ __constant__ uchar4 dev_centers[500];

__device__ int calc_best_distance(uchar4 point, int* affiliation, int n)
{
    int best_class = 255;
    float best_distance = 450; // sqrt(255*255 * 3) = 441.6729559300637

    float curr_distance;
    for (int i = 0; i < n; i++) {
        curr_distance = distance(point, dev_centers[i]);
        if (curr_distance < best_distance) {
            best_class = i;
            best_distance = curr_distance;
        }
    }

    return best_class;
}

// классификация пикселей по текущим центрам групп.
__global__ void classify(
    uchar4* data, int n, int* affiliation, int n_classes)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = id; i < n; i += offset) {
        int best = calc_best_distance(data[i], affiliation, n_classes);
        affiliation[i] = best;
        data[i].w = best;
    }
}

// сравнение двух массивов классификации пикселей.
__global__ void compare_arrays(int* prev, int* next, unsigned long long* result, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for (int i = id; i < n; i += offset) {
        if (prev[i] != next[i]) {
            atomicAdd(result, 1);
        }
    }
}

// вычисление новых центров классов
__global__ void identify_centers(
    uchar4* data, int* affiliations, int n,
    uchar4* new_centers, uint3* cache, int* cache_count, int n_classes)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    // суммирование значений пикселей по классам
    uchar4 elem;
    int affiliation;
    for (int i = id; i < n; i += offset) {
        affiliation = affiliations[i];
        elem = data[i];

        uint3 center = cache[affiliation];
        center.x += int(elem.x);
        center.y += int(elem.y);
        center.z += int(elem.z);
        cache[affiliation] = center;

        cache_count[affiliation]++;
    }

    __syncthreads();
    // присваивание новых значений центров классов.
    for (int i = id; i < n_classes; i += offset) {
        uint3 cache_elem = cache[i];
        int count = cache_count[i];
        cache_elem.x /= count;
        cache_elem.y /= count;
        cache_elem.z /= count;
        uchar4 elem;
        elem.x = cache_elem.x;
        elem.y = cache_elem.y;
        elem.z = cache_elem.z;
        new_centers[i] = elem;
    }
}

typedef struct {
    int x, y;
} Center;

void launch_k_means(uchar4* data, const int w, const int h, const Center* start_centers, const int n_classes)
{
    const int n = h * w;

    uchar4* dev_next_centers;
    uchar4* dev_data;
    uint3* dev_cache;
    int* dev_cache_count;
    int
        *dev_prev_affiliation,
        *dev_next_affiliation;

    CSC(cudaMalloc(&dev_next_centers, sizeof(uchar4) * n));
    CSC(cudaMalloc(&dev_cache, sizeof(uint3) * n));
    CSC(cudaMalloc(&dev_cache_count, sizeof(int) * n));

    CSC(cudaMalloc(&dev_next_affiliation, sizeof(int) * n));
    CSC(cudaMalloc(&dev_prev_affiliation, sizeof(int) * n));
    // ?
    CSC(cudaMemset(dev_prev_affiliation, 255, sizeof(int) * n));

    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * n));
    CSC(cudaMemcpy(dev_data, data, n, cudaMemcpyHostToDevice));

    {
        uchar4* tmp_centers;
        // значения указанных пикселей.
        tmp_centers = (uchar4*)malloc(sizeof(uchar4) * n);
        for (int i = 0; i < n; i++) {
            tmp_centers[i] = data[start_centers[i].x * w + start_centers[i].y];
        }
        CSC(cudaMemcpyToSymbol(dev_centers, tmp_centers, sizeof(uchar4) * n));
        free(tmp_centers);
    }

    int i = 0;
    while (i != 1) {
        i = 1;
        // вычисление принадлежности к классам
        START_KERNEL(
            blocks, threads, classify,
            data, n, dev_next_affiliation, n_classes)

        // сравнение предыдущей классификации и текущей
        unsigned long long
            equal = 0,
            *dev_equal;
        CSC(cudaMalloc(&dev_equal, sizeof(unsigned long long)));

        CSC(cudaMemset(dev_equal, 0, sizeof(unsigned long long)));
        START_KERNEL(
            blocks, threads, compare_arrays,
            dev_prev_affiliation, dev_next_affiliation, dev_equal, n)
        CSC(cudaMemcpy(&equal, dev_equal, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (equal == 0) {
            break;
        }
        CSC(cudaFree(dev_equal));

        // вычисление новых центров классов
        int* tmp = dev_prev_affiliation;
        dev_prev_affiliation = dev_next_affiliation;
        dev_next_affiliation = tmp;
        CSC(cudaMemset(dev_cache, 0, sizeof(uint3) * n));
        CSC(cudaMemset(dev_cache_count, 0, sizeof(int) * n));
        START_KERNEL(
            blocks, threads, identify_centers,
            data, dev_next_affiliation, n,
            dev_next_centers, dev_cache, dev_cache_count, n_classes)

        CSC(cudaMemcpyToSymbol(dev_centers, dev_next_centers, sizeof(uchar4) * n, 0, cudaMemcpyDeviceToDevice));
    }

    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_next_affiliation));
    CSC(cudaFree(dev_prev_affiliation));
    CSC(cudaFree(dev_next_centers));
    CSC(cudaFree(dev_cache));
    CSC(cudaFree(dev_cache_count));
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