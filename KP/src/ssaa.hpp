#ifndef SSAA_HPP
#define SSAA_HPP

#include "helpers.cuh"

void ssaa_omp(uchar4* dst, uchar4* src, int new_w, int new_h, int w, int h);

__global__ void ssaa_kernel(
    uchar4* dst, uchar4* src,
    int new_w, int new_h,
    int w, int h);

#endif
