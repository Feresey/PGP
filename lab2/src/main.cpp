#include <cmath>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "serialize.h"

template <class T>
struct left4dead {
    T x;
    T y;
    T z;
    T w;
};

typedef unsigned long long ull;
typedef unsigned char uchar;

typedef left4dead<uchar> uchar4;
typedef left4dead<float> float4;
typedef left4dead<int> int4;
typedef left4dead<ull> ulonglong4;

float4 make_float4(float x, float y, float z, float w)
{
    return float4 { x, y, z, w };
}

uchar4 make_uchar4(uchar x, uchar y, uchar z, uchar w)
{
    return uchar4 { x, y, z, w };
}

struct texture {
    size_t w;
    size_t h;
    uchar4* data;
};

uchar4 tex2D(const texture& tex, int x, int y)
{
    if (x < 0) {
        x = 0;
    }
    if (x >= tex.w) {
        x = int(tex.w) - 1;
    }
    if (y < 0) {
        y = 0;
    }
    if (y >= tex.h) {
        y = int(tex.h) - 1;
    }
    return tex.data[size_t(y) * tex.w + size_t(x)];
}

texture tex;

int4 uchar4sum(uchar4 a, uchar4 b, uchar4 c)
{
    int4 res;
    res.x = int(a.x) + int(b.x) + int(c.x);
    res.y = int(a.y) + int(b.y) + int(c.y);
    res.z = int(a.z) + int(b.z) + int(c.z);
    res.w = 0;
    return res;
}

int4 int4sub(int4 a, int4 b)
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

float prewitt(uchar4* z)
{
    int4 up = uchar4sum(z[0], z[1], z[2]);
    int4 down = uchar4sum(z[6], z[7], z[8]);

    int4 right = uchar4sum(z[0], z[3], z[6]);
    int4 left = uchar4sum(z[2], z[5], z[8]);

    int4 g_x = int4sub(down, up);
    int4 g_y = int4sub(left, right);

    return meanless(norm(g_x), norm(g_y));
}

void kernel(uchar4* out, uint32_t w, uint32_t h)
{
    uchar4 z[9];
    int left, right, top, bottom;
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {

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
            uchar res_byte = uchar(res);
            if (res < 0) {
                printf("ERROR: ты обосрался: %f\n", res);
            }
            if (res > 255) {
                res_byte = 255;
            }
            out[uint32_t(x) + uint32_t(y) * w] = make_uchar4(res_byte, res_byte, res_byte, 0);
        }
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

    tex.w = w;
    tex.h = h;
    tex.data = (uchar4*)data;

    uchar4* res_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    struct timespec start_time = timer_start();
    kernel(res_data, w, h);
    fprintf(stderr, "kernel time: %ld\n", timer_end(start_time));

    FILE* out = fopen(output, "wb");
    err = write_image(out, (uint32_t*)res_data, w, h);
    if (err != 0) {
        printf("ERROR in %s:%d write image: %d", __FILE__, __LINE__, err);
    }
    fclose(out);

    free(data);
    free(res_data);
}
