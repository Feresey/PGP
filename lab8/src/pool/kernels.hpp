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

dim3_type layer_tag_to_dim3_type(layer_tag tag);
layer_tag dim3_type_to_layer_tag(dim3_type type);

class DeviceProblem {
    const BlockGrid& grid;
    const int n_devices;

    void get_border(
        std::vector<double>& out, std::vector<double> data,
        int a_szie, int b_size,
        int border_idx, layer_tag tag);

    void set_border(
        std::vector<double>& dest, std::vector<double> data,
        int a_szie, int b_size,
        int border_idx, layer_tag tag);

    void set_device(int) const;

public:
    const int kernel_grid_dim, kernel_block_dim;
    DeviceProblem(const BlockGrid& grid, const int n_devices, const int kernel_grid_dim = 8, const int kernel_block_dim = 8);

    void compute(std::vector<double>& out, std::vector<double>& data, mydim3<double> height);
    void calc_abs_error(std::vector<double>& out, std::vector<double>& data);
};

#endif
