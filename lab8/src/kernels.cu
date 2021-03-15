#include <cmath>
#include <vector>

#include <thrust/device_vector.h>

#include "dim3/dim3.hpp"
#include "grid/grid.hpp"
#include "helpers.cuh"
#include "kernels.hpp"

#define BORDER_GRID_DIM 32
#define BORDER_BLOCK_DIM 32

#define BORDER_DIMS dim3(BORDER_BLOCK_DIM, BORDER_BLOCK_DIM), dim3(BORDER_GRID_DIM, BORDER_GRID_DIM)

/* KERNELS */

__global__ void get_border_kernel(
    double* out, double* buf,
    Grid grid, int a_size, int b_size,
    int border_idx, layer_tag tag)
{
    const int id_a = threadIdx.x + blockIdx.x * blockDim.x,
              id_b = threadIdx.y + blockIdx.y * blockDim.y,
              offset_a = blockDim.x * gridDim.x,
              offset_b = blockDim.y * gridDim.y;

    double temp;

    for (int b = id_b; b < b_size; b += offset_b) {
        for (int a = id_a; a < a_size; a += offset_a) {
            switch (tag) {
            case LEFT_RIGHT:
                temp = buf[grid.cell_absolute_id(border_idx, a, b)];
                break;
            case FRONT_BACK:
                temp = buf[grid.cell_absolute_id(a, border_idx, b)];
                break;
            case TOP_BOTTOM:
                temp = buf[grid.cell_absolute_id(a, b, border_idx)];
                break;
            }

            out[b * a_size + a] = temp;
        }
    }
}

__global__ void set_border_kernel(
    double* dest, double* buf,
    Grid grid, int a_size, int b_size,
    int border_idx,
    layer_tag tag)
{
    const int id_a = threadIdx.x + blockIdx.x * blockDim.x,
              id_b = threadIdx.y + blockIdx.y * blockDim.y,
              offset_a = blockDim.x * gridDim.x,
              offset_b = blockDim.y * gridDim.y;

    int dest_idx;

    for (int b = id_b; b < b_size; b += offset_b) {
        for (int a = id_a; a < a_size; a += offset_a) {
            switch (tag) {
            case LEFT_RIGHT:
                dest_idx = grid.cell_absolute_id(border_idx, a, b);
                break;
            case FRONT_BACK:
                dest_idx = grid.cell_absolute_id(a, border_idx, b);
                break;
            case TOP_BOTTOM:
                dest_idx = grid.cell_absolute_id(a, b, border_idx);
                break;
            }

            dest[dest_idx] = buf[b * a_size + a];
        }
    }
}

__global__ void compute_kernel(
    double* out, double* data,
    Grid grid,
    mydim3<int> bsize,
    mydim3<double> h)
{
    const int id_x = threadIdx.x + blockIdx.x * blockDim.x,
              id_y = threadIdx.y + blockIdx.y * blockDim.y,
              id_z = threadIdx.z + blockIdx.z * blockDim.z,
              offset_x = blockDim.x * gridDim.x,
              offset_y = blockDim.y * gridDim.y,
              offset_z = blockDim.z * gridDim.z;

    const double inv_hx = 1.0 / (h.x * h.x),
                 inv_hy = 1.0 / (h.y * h.y),
                 inv_hz = 1.0 / (h.z * h.z);

    for (int i = id_x; i < bsize.x; i += offset_x) {
        for (int j = id_y; j < bsize.y; j += offset_y) {
            for (int k = id_z; k < bsize.z; k += offset_z) {
                double num = 0.0
                    + (data[grid.cell_absolute_id(i + 1, j, k)] + data[grid.cell_absolute_id(i - 1, j, k)]) * inv_hx
                    + (data[grid.cell_absolute_id(i, j + 1, k)] + data[grid.cell_absolute_id(i, j - 1, k)]) * inv_hy
                    + (data[grid.cell_absolute_id(i, j, k + 1)] + data[grid.cell_absolute_id(i, j, k - 1)]) * inv_hz;
                double denum = 2.0 * (inv_hx + inv_hy + inv_hz);

                out[grid.cell_absolute_id(i, j, k)] = num / denum;
            }
        }
    }
}

__global__ void abs_error_kernel(double* out, double* data, Grid grid, mydim3<int> bsize)
{
    const int id_x = threadIdx.x + blockIdx.x * blockDim.x,
              id_y = threadIdx.y + blockIdx.y * blockDim.y,
              id_z = threadIdx.z + blockIdx.z * blockDim.z,
              offset_x = blockDim.x * gridDim.x,
              offset_y = blockDim.y * gridDim.y,
              offset_z = blockDim.z * gridDim.z;

    for (int i = id_x - 1; i <= bsize.x; i += offset_x) {
        for (int j = id_y - 1; j <= bsize.y; j += offset_y) {
            for (int k = id_z - 1; k <= bsize.z; k += offset_z) {
                out[grid.cell_absolute_id(i, j, k)] = fabsf(out[grid.cell_absolute_id(i, j, k)] - data[grid.cell_absolute_id(i, j, k)]);
            }
        }
    }
}

/* METHODS */

void DeviceProblem::get_border(
    thrust::device_vector<double>& out, thrust::device_vector<double> data,
    int a_size, int b_size,
    int border_idx, layer_tag tag)
{
    double *out_raw = out.data().get(),
           *data_raw = data.data().get();
    START_KERNEL((get_border_kernel<<<BORDER_DIMS>>>(out_raw, data_raw, grid, a_size, b_size, border_idx, tag)));
}

void DeviceProblem::set_border(
    thrust::device_vector<double>& dest, thrust::device_vector<double> data,
    int a_size, int b_size,
    int border_idx, layer_tag tag)
{
    double *dest_raw = dest.data().get(),
           *data_raw = data.data().get();
    START_KERNEL((set_border_kernel<<<BORDER_DIMS>>>(dest_raw, data_raw, grid, a_size, b_size, border_idx, tag)));
}

dim3 get_grid_dim(const DeviceProblem* dp)
{
    return dim3(dp->kernel_grid_dim, dp->kernel_grid_dim, dp->kernel_grid_dim);
}

dim3 get_block_dim(const DeviceProblem* dp)
{
    return dim3(dp->kernel_block_dim, dp->kernel_block_dim, dp->kernel_block_dim);
}

void DeviceProblem::compute(
    thrust::device_vector<double>& out,
    thrust::device_vector<double>& data,
    mydim3<double> height)
{
    double *out_raw = out.data().get(),
           *data_raw = data.data().get();

    START_KERNEL((
        compute_kernel<<<get_grid_dim(this), get_block_dim(this)>>>(
            out_raw, data_raw, grid, grid.bsize, height)));
}

void DeviceProblem::calc_abs_error(
    thrust::device_vector<double>& out,
    thrust::device_vector<double>& data)
{
    double *out_raw = out.data().get(),
           *data_raw = data.data().get();

    START_KERNEL((
        abs_error_kernel<<<get_grid_dim(this), get_block_dim(this)>>>(
            out_raw, data_raw, grid, grid.bsize)));
}
