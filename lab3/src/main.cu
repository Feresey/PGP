
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.cuh"

#define EPS 1e-3

//| да, костыль|
int blocks = 1;
int threads = 32;

__device__ __constant__ float4 dev_centers[500];

//| расстояние между центром и пикселем|
__device__ float distance(const uchar4& a, const float4& b)
{
    float x = b.x - float(a.x),
          y = b.y - float(a.y),
          z = b.z - float(a.z);
    return x * x + y * y + z * z;
}

//| близость двух центров классов|
__device__ float norm(float4 a, float4 b)
{
    float x = a.x - b.x,
          y = a.y - b.y,
          z = a.z - b.z;
    return x * x + y * y + z * z;
}

__device__ int calc_best_distance(const uchar4& point, const int n_classes)
{
    int best_class = 0;
    float best_distance = 1e15;

    for (int i = 0; i < n_classes; i++) {
        float curr_distance = distance(point, dev_centers[i]);
        if (curr_distance < best_distance) {
            best_class = i;
            best_distance = curr_distance;
        }
    }

    return best_class;
}

//| классификация пикселей по текущим центрам групп.|
__global__ void kernel(
    uchar4* data, size_t n,
    ulonglong4* cache, uint32_t n_classes)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    for (size_t i = id; i < n; i += offset) {
        uchar4& elem = data[i];
        elem.w = calc_best_distance(elem, n_classes);

        //| вычисление новых центров классов|
        //| суммирование значений пикселей по классам|
        ulonglong4* cache_elem = &cache[elem.w];
        atomicAdd(&cache_elem->x, elem.x);
        atomicAdd(&cache_elem->y, elem.y);
        atomicAdd(&cache_elem->z, elem.z);
        atomicAdd(&cache_elem->w, 1);
    }
}

__global__ void calc_centers(
    float4* new_centers, ulonglong4* cache, uint32_t n_classes,
    unsigned long long* equal)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    //| присваивание новых значений центров классов.|
    for (uint32_t i = id; i < n_classes; i += offset) {
        ulonglong4 cache_elem = cache[i];
        float4 elem = make_float4(
            float(cache_elem.x) / float(cache_elem.w),
            float(cache_elem.y) / float(cache_elem.w),
            float(cache_elem.z) / float(cache_elem.w),
            0.0f);
        new_centers[i] = elem;

        //| условие сходимости -- центры не изменились|
        float4 old = dev_centers[i];
        float diff = norm(old, elem);
        if (diff > EPS) {
            atomicAdd(equal, 1);
        }
    }
}

typedef struct {
    int x, y;
} Center;

void launch_k_means(uchar4* host_data, const size_t w, const size_t h, const Center* start_centers, const uint32_t n_classes)
{
    const size_t n = h * w;

    uchar4* dev_data;
    //| свежевычисленные центры на основе текущего распределения по классам|
    float4* dev_next_centers;
    //| суммируются все пиксели для вычисления центров классов|
    ulonglong4* dev_cache;

    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * n));
    CSC(cudaMemcpy(dev_data, host_data, sizeof(uchar4) * n, cudaMemcpyHostToDevice));

    CSC(cudaMalloc(&dev_next_centers, sizeof(float4) * n_classes));
    CSC(cudaMalloc(&dev_cache, sizeof(ulonglong4) * n_classes));

    //| инициализация центров классов по их координатам|
    {
        float4* tmp_centers;
        //| значения указанных пикселей.|
        tmp_centers = (float4*)malloc(sizeof(float4) * n_classes);
        for (uint32_t i = 0; i < n_classes; i++) {
            uchar4 elem = host_data[(size_t)start_centers[i].y * w + (size_t)start_centers[i].x];
            tmp_centers[i] = make_float4(elem.x, elem.y, elem.z, 0.0f);
        }
        CSC(cudaMemcpy(dev_next_centers, tmp_centers, sizeof(float4) * n_classes, cudaMemcpyHostToDevice));
        free(tmp_centers);
    }

    unsigned long long equal = 1,
                       *dev_equal;
    CSC(cudaMalloc(&dev_equal, sizeof(unsigned long long)));

    while (equal != 0) {
        CSC(cudaMemcpyToSymbol(dev_centers, dev_next_centers, sizeof(float4) * n_classes, 0, cudaMemcpyDeviceToDevice));

        CSC(cudaMemset(dev_equal, 0, sizeof(unsigned long long)));
        CSC(cudaMemset(dev_cache, 0, sizeof(ulonglong4) * n_classes));

        START_KERNEL((kernel<<<dim3(blocks, 1, 1), dim3(threads, 1, 1)>>>(
            dev_data, n, dev_cache, n_classes)));

        START_KERNEL((calc_centers<<<1, n_classes>>>(
            dev_next_centers, dev_cache, n_classes,
            dev_equal)));

        CSC(cudaMemcpy(&equal, dev_equal, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }

    CSC(cudaFree(dev_equal));
    CSC(cudaMemcpy(host_data, dev_data, sizeof(uchar4) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_next_centers));
    CSC(cudaFree(dev_cache));
}

#include <time.h>

// call this function to start a nanosecond-resolution timer
struct timespec timer_start()
{
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

// call this function to end a timer, returning nanoseconds elapsed as a long
unsigned long long timer_end(struct timespec start_time)
{
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    unsigned long long diffInNanos = (unsigned long long)(end_time.tv_sec - start_time.tv_sec) * (unsigned long long)1e9 + (unsigned long long)(end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}

#ifdef BENCHMARK
int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-blocks") == 0) {
            blocks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-threads") == 0) {
            threads = atoi(argv[++i]);
        }
    }
#else
int main()
#endif

    char input[PATH_MAX], output[PATH_MAX];

    scanf("%s", input);
    scanf("%s", output);

    uint32_t n_classes;
    Center* centers;
    scanf("%d", &n_classes);
    centers = (Center*)malloc(n_classes * sizeof(Center));
    for (uint32_t i = 0; i < n_classes; i++) {
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
    read_image(in, &data, &w, &h);
    fclose(in);

    fprintf(stderr, "blocks = %d\nthreads = %d\n", blocks, threads);

    struct timespec start_time = timer_start();
    launch_k_means(data, w, h, centers, n_classes);
    fprintf(stderr, "run: %lld\n", timer_end(start_time));

    FILE* out = fopen(output, "wb");
    write_image(out, data, w, h);
    fclose(out);

    free(data);
    free(centers);
    return 0;
}