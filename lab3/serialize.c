#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(SIZE_EQ, WANT)    \
    {                           \
        size_t err = (SIZE_EQ); \
        if (err != WANT) {      \
            return err;         \
        };                      \
    }

typedef unsigned char uchar;

int read_image(FILE* f, uchar** out, uint32_t* w, uint32_t* h)
{
    CHECK(fread(w, sizeof(uint32_t), 1, f), 1);
    CHECK(fread(h, sizeof(uint32_t), 1, f), 1);
    const size_t size = *w * *h;
    *out = (uint32_t*)malloc(size * 4);
    CHECK(fread(*out, 1, size * 4, f), size * 4);
    return 0;
}

int write_image(FILE* f, uchar* data, uint32_t w, uint32_t h)
{
    CHECK(fwrite(&w, sizeof(uint32_t), 1, f), 1);
    CHECK(fwrite(&h, sizeof(uint32_t), 1, f), 1);
    CHECK(fwrite(data, 1, w * h * 4, f), w * h * 4);
    return 0;
}