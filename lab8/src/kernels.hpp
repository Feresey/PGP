#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <thrust/device_vector.h>

#include "dim3/dim3.hpp"
#include "grid/grid.hpp"

enum side_tag {
    LEFT,
    RIGHT,
    FRONT,
    BACK,
    TOP,
    BOTTOM
};

enum layer_tag {
    TOP_BOTTOM,
    LEFT_RIGHT,
    FRONT_BACK
};

class DeviceProblem {
    const Grid& grid;

public:
    const int kernel_grid_dim, kernel_block_dim;
    DeviceProblem(const Grid& grid, const int kernel_grid_dim = 8, const int kernel_block_dim = 8);

    void get_border(
        thrust::device_vector<double>& out, thrust::device_vector<double> data,
        int a_szie, int b_size,
        int border_idx, layer_tag tag);

    void set_border(
        thrust::device_vector<double>& dest, thrust::device_vector<double> data,
        int a_szie, int b_size,
        int border_idx, layer_tag tag);

    void compute(
        thrust::device_vector<double>& out,
        thrust::device_vector<double>& data,
        mydim3<double> height);

    void calc_abs_error(
        thrust::device_vector<double>& out,
        thrust::device_vector<double>& data);
};

#endif
