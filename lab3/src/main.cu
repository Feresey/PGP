#include <stdio.h>

#include "helpers.cuh"

#define MAX_CLUSTERS 32
#define EPS 0.01f

typedef unsigned long long ull;

// да, костыль
dim3 blocks = 1;
dim3 threads = 32;

__device__ __constant__ float4 dev_centers[MAX_CLUSTERS];

__device__ float dev_dist(uchar4 p, int n_cluster)
{
    float dx = float(p.x) - dev_centers[n_cluster].x;
    float dy = float(p.y) - dev_centers[n_cluster].y;
    float dz = float(p.z) - dev_centers[n_cluster].z;

    return dx * dx + dy * dy + dz * dz;
}

float dist(float4 a, float4 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    return dx * dx + dy * dy + dz * dz;
}

__global__ void calc_distances(uchar4* pixels, size_t n_pixels, uint32_t n_clusters)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n_pixels; i += offset) {
        int nearest_cluster_index = 0;
        float min_dist = dev_dist(pixels[i], 0);
        for (int j = 1; j < n_clusters; ++j) {
            float temp_dist = dev_dist(pixels[i], j);
            if (temp_dist < min_dist) {
                nearest_cluster_index = j;
                min_dist = temp_dist;
            }
        }
        pixels[i].w = nearest_cluster_index;
    }
}

typedef struct {
    int x, y;
} Center;

void launch_k_means(
    uchar4* pixels, const size_t w, const size_t h,
    const Center* cluster_centers, const uint32_t n_clusters)
{
    const size_t n_pixels = h * w;

    uchar4* dev_pixels;
    CSC(cudaMalloc(&dev_pixels, sizeof(uchar4) * n_pixels));

    CSC(cudaMemcpy(dev_pixels, pixels, sizeof(uchar4) * n_pixels, cudaMemcpyHostToDevice));

    ulonglong4 color_by_cluster_idx[MAX_CLUSTERS];
    float4 host_centers[MAX_CLUSTERS];

    for (uint32_t i = 0; i < n_clusters; ++i) {
        const Center center = cluster_centers[i];
        uchar4 center_pixel = pixels[(size_t)center.y * w + (size_t)center.x];
        host_centers[i].x = center_pixel.x;
        host_centers[i].y = center_pixel.y;
        host_centers[i].z = center_pixel.z;
        host_centers[i].w = 0.0f;
    }

    while (true) {
        CSC(cudaMemcpyToSymbol(dev_centers, host_centers, sizeof(float4) * n_clusters, 0, cudaMemcpyHostToDevice));

        START_KERNEL((calc_distances<<<blocks, threads>>>(dev_pixels, n_pixels, n_clusters)));

        CSC(cudaMemcpy(pixels, dev_pixels, sizeof(uchar4) * n_pixels, cudaMemcpyDeviceToHost));

        memset(color_by_cluster_idx, 0, sizeof(ulonglong4) * MAX_CLUSTERS);
        for (size_t i = 0; i < n_pixels; ++i) {
            uchar4 cur_pixel = pixels[i];
            color_by_cluster_idx[cur_pixel.w].x += ull(cur_pixel.x);
            color_by_cluster_idx[cur_pixel.w].y += ull(cur_pixel.y);
            color_by_cluster_idx[cur_pixel.w].z += ull(cur_pixel.z);
            ++color_by_cluster_idx[cur_pixel.w].w;
        }

        bool in_eps = true;

        for (uint32_t i = 0; i < n_clusters; ++i) {
            ull cluster_count = color_by_cluster_idx[i].w;
            ulonglong4 cluster_color = color_by_cluster_idx[i];
            // printf("%i: %llu %llu %llu (%llu)\n", i,cluster_color.x, cluster_color.y, cluster_color.z, cluster_count);
            float4 temp = make_float4(
                float(cluster_color.x) / float(cluster_count),
                float(cluster_color.y) / float(cluster_count),
                float(cluster_color.z) / float(cluster_count),
                0.0f);

            if (dist(host_centers[i], temp) > EPS) {
                in_eps = false;
            }

            host_centers[i] = temp;
        }

        if (in_eps) {
            break;
        }
    }

    CSC(cudaFree(dev_pixels));
}

int main()
{
    char input[255], output[255];

    scanf("%s", input);
    scanf("%s", output);

    uint32_t n_clusters;
    Center* centers;
    scanf("%d", &n_clusters);
    if (n_clusters == 0) {
        exit(2);
    }
    centers = (Center*)malloc(n_clusters * sizeof(Center));
    for (uint32_t i = 0; i < n_clusters; i++) {
        scanf("%d", &centers[i].x);
        scanf("%d", &centers[i].y);
    }

    FILE* in = fopen(input, "rb");
    if (in == NULL || ferror(in)) {
        perror(NULL);
        fprintf(stderr, "ERROR opening input file: %s\n", input);
        exit(0);
    }

    uchar4* data;
    uint32_t w, h;
    read_image(in, &data, &w, &h);
    fclose(in);

    launch_k_means(data, w, h, centers, n_clusters);

    FILE* out = fopen(output, "wb");
    if (out == NULL || ferror(out)) {
        perror(NULL);
        fprintf(stderr, "ERROR opening output file: %s\n", output);
        exit(0);
    }
    write_image(out, data, w, h);
    fclose(out);

    free(data);
    free(centers);
    return 0;
}