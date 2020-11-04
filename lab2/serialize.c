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