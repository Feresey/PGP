#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CLUSTERS 32
#define EPS 0.01f

typedef unsigned long long ull;
typedef unsigned char uchar;

template <class T>
struct left4dead {
    T x;
    T y;
    T z;
    T w;
};

typedef left4dead<uchar> uchar4;
typedef left4dead<float> float4;
typedef left4dead<ull> ulonglong4;

// need types above
#include "helpers.cuh"

float4 make_float4(float x, float y, float z, float w)
{
    return float4 { x, y, z, w };
}

uchar4 make_uchar4(uchar x, uchar y, uchar z, uchar w)
{
    return uchar4 { x, y, z, w };
}

float4 dev_centers[MAX_CLUSTERS];

float pixel_dist(uchar4 a, float4 b)
{
    float dx = float(a.x) - b.x;
    float dy = float(a.y) - b.y;
    float dz = float(a.z) - b.z;
    return dx * dx + dy * dy + dz * dz;
}

float center_dist(float4 a, float4 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    return dx * dx + dy * dy + dz * dz;
}

void calc_distances(uchar4* pixels, size_t n_pixels, uint32_t n_clusters)
{
    for (size_t i = 0; i < n_pixels; ++i) {
        uchar nearest_cluster_index = 0;
        float min_dist = pixel_dist(pixels[i], dev_centers[0]);
        for (uchar j = 1; j < n_clusters; ++j) {
            float temp_dist = pixel_dist(pixels[i], dev_centers[j]);
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
    char* testname,
    uchar4* pixels, const size_t w, const size_t h,
    const Center* cluster_centers, const uint32_t n_clusters)
{
    const size_t n_pixels = w * h;

    ulonglong4 color_by_cluster_idx[MAX_CLUSTERS];
    float4 host_centers[MAX_CLUSTERS];

    for (uint32_t i = 0; i < n_clusters; ++i) {
        const Center center = cluster_centers[i];
        const size_t pos = (size_t)center.y * w + (size_t)center.x;
        uchar4 center_pixel = pixels[pos];
        host_centers[i].x = center_pixel.x;
        host_centers[i].y = center_pixel.y;
        host_centers[i].z = center_pixel.z;
        host_centers[i].w = 0.0f;

        // printf("%d: (x=%d, y=%d, pos=%ld) %02x%02x%02x\n",
        //     i,
        //     center.x, center.y, pos,
        //     center_pixel.x, center_pixel.y, center_pixel.z);
    }

    int iter = 0;
    while (++iter) {

        // char outname[256];
        // sprintf(outname, "%s.%d.points", testname, iter);
        // FILE* out = fopen(outname, "w");
        // for (size_t i = 0; i < n_pixels; ++i) {
        //     const uchar4 pixel = pixels[i];
        //     const float4 cluster = host_centers[pixel.w];
        //     // fprintf(out, "%lu %d %d %d %d %d %d %d\n", i, pixel.x, pixel.y, pixel.z, pixel.w, uchar(cluster.x), uchar(cluster.y), uchar(cluster.z));
        // }
        // fclose(out);

        memcpy(dev_centers, host_centers, sizeof(float4) * n_clusters);
        calc_distances(pixels, n_pixels, n_clusters);

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

            if (center_dist(host_centers[i], temp) > EPS) {
                in_eps = false;
            }

            host_centers[i] = temp;
        }

        if (in_eps) {
            break;
        }
    }

    for (size_t i = 0; i < n_pixels; ++i) {
        uchar4 p = pixels[i];
        float4 cluster = host_centers[p.w];
        pixels[i] = make_uchar4(uchar(cluster.x), uchar(cluster.y), uchar(cluster.z), 255);
    }
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
long timer_end(struct timespec start_time)
{
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
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

    struct timespec start_time = timer_start();
    launch_k_means(input, data, w, h, centers, n_clusters);
    fprintf(stderr, "run: %d\n", timer_end(start_time));

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