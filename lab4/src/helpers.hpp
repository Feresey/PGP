#ifndef HELPERS_HPP
#define HELPERS_HPP

#ifdef __cplusplus

#include <vector>

typedef std::vector<double> host_matrix;
typedef std::vector<double> dev_matrix;

#define __host__
#define __device__
#define __global__
#define __shared__

struct dim3 {
    uint x, y, z;
    dim3()
        : x(0)
        , y(0)
        , z(0)
    {
    }
};

static const dim3 threadIdx, blockIdx, blockDim, gridDim;

void __syncthreads();

#endif
#endif