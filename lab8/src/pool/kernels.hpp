#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <vector>

#include "dim3/dim3.hpp"
#include "grid/grid.hpp"

enum side_tag {
    LEFT = 1 << 0,
    RIGHT = 1 << 1,
    FRONT = 1 << 2,
    BACK = 1 << 3,
    TOP = 1 << 4,
    BOTTOM = 1 << 5
};

enum layer_tag {
    VERTICAL = TOP | BOTTOM,
    LEFT_RIGHT = LEFT | RIGHT,
    FRONT_BACK = FRONT | BACK
};

dim3_type side_tag_to_dim3_type(side_tag tag);
dim3_type layer_tag_to_dim3_type(layer_tag tag);
layer_tag dim3_type_to_layer_tag(dim3_type type);
layer_tag side_tag_to_layer_tag(side_tag tag);

class DeviceKernels {
    BlockGrid grid;
    int kernel_block_dim, kernel_grid_dim;

protected:
    void get_border(
        double* out, double* data,
        int a_szie, int b_size,
        int border_idx, layer_tag tag);

    void set_border(
        double* dest, double* data,
        int a_szie, int b_size,
        int border_idx, layer_tag tag);

    double compute(double* out, double* data, mydim3<double> height);

public:
    DeviceKernels(BlockGrid grid, int kernel_grid_dim = 8, int kernel_block_dim = 8);
};

#endif
