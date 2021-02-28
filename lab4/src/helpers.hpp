#ifndef HELPERS_HPP
#define HELPERS_HPP

#ifdef __cplusplus

#include <vector>
#include <vector_types.h>

typedef std::vector<double> host_matrix;
typedef std::vector<double> dev_matrix;

#define START_KERNEL(X)
#define CSC(X)

#define __host__
#define __device__
#define __global__
#define __shared__

static const dim3 threadIdx, blockIdx, blockDim, gridDim;

void __syncthreads() { }

#endif
#endif