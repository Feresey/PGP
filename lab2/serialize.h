#ifndef SERIALIZE_H
#define SERIALIZE_H

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int read_image(FILE* f, uint32_t** out, uint32_t* w, uint32_t* h);
int write_image(FILE* f, uint32_t* data, uint32_t w, uint32_t h);

#ifdef __cplusplus
};
#endif

#endif
