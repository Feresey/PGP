#include <cmath>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "dim3/dim3.hpp"
#include "grid/grid.hpp"
#include "helpers.cuh"
#include "kernels.hpp"

#define BORDER_GRID_DIM 32
#define BORDER_BLOCK_DIM 32

#define BORDER_DIMS_2D dim3(BORDER_BLOCK_DIM, BORDER_BLOCK_DIM), dim3(BORDER_GRID_DIM, BORDER_GRID_DIM)
#define BORDER_DIMS_3D(BLOCK_DIM, GRID_DIM) dim3(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM), dim3(GRID_DIM, GRID_DIM, GRID_DIM)

/* KERNELS */

__global__ void get_border_kernel(
    double* out, double* buf,
    BlockGrid grid, int a_size, int b_size,
    int border_idx, layer_tag tag)
{
    const int id_a = threadIdx.x + blockIdx.x * blockDim.x,
              id_b = threadIdx.y + blockIdx.y * blockDim.y,
              offset_a = blockDim.x * gridDim.x,
              offset_b = blockDim.y * gridDim.y;

    double temp;

    for (int a = id_b; a < a_size; a += offset_a) {
        for (int b = id_a; b < b_size; b += offset_b) {
            switch (tag) {
            case LEFT_RIGHT:
                temp = buf[grid.cell_absolute_id(border_idx, b, a)];
                break;
            case FRONT_BACK:
                temp = buf[grid.cell_absolute_id(b, border_idx, a)];
                break;
            case VERTICAL:
                temp = buf[grid.cell_absolute_id(b, a, border_idx)];
                break;
            }

            out[a * b_size + b] = temp;
        }
    }
}

__global__ void set_border_kernel(
    double* dest, double* buf,
    BlockGrid grid, int a_size, int b_size,
    int border_idx,
    layer_tag tag)
{
    const int id_a = threadIdx.x + blockIdx.x * blockDim.x,
              id_b = threadIdx.y + blockIdx.y * blockDim.y,
              offset_a = blockDim.x * gridDim.x,
              offset_b = blockDim.y * gridDim.y;

    int dest_idx;

    for (int a = id_b; a < a_size; a += offset_a) {
        for (int b = id_a; b < b_size; b += offset_b) {
            switch (tag) {
            case LEFT_RIGHT:
                dest_idx = grid.cell_absolute_id(border_idx, b, a);
                break;
            case FRONT_BACK:
                dest_idx = grid.cell_absolute_id(b, border_idx, a);
                break;
            case VERTICAL:
                dest_idx = grid.cell_absolute_id(b, a, border_idx);
                break;
            }

            dest[dest_idx] = buf[a * b_size + b];
        }
    }
}

__global__ void compute_kernel(
    double* out, double* data,
    BlockGrid grid,
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

    for (int i = id_x; i < grid.bsize.x; i += offset_x) {
        for (int j = id_y; j < grid.bsize.y; j += offset_y) {
            for (int k = id_z; k < grid.bsize.z; k += offset_z) {
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

__global__ void abs_error_kernel(double* out, double* data, BlockGrid grid)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x,
              idy = threadIdx.y + blockIdx.y * blockDim.y,
              idz = threadIdx.z + blockIdx.z * blockDim.z,
              offset_x = blockDim.x * gridDim.x,
              offset_y = blockDim.y * gridDim.y,
              offset_z = blockDim.z * gridDim.z;

    for (int i = idx - 1; i <= grid.bsize.x; i += offset_x) {
        for (int j = idy - 1; j <= grid.bsize.y; j += offset_y) {
            for (int k = idz - 1; k <= grid.bsize.z; k += offset_z) {
                int cell_id = grid.cell_absolute_id(i, j, k);
                if (i == -1 || j == -1 || k == -1
                    || i == grid.bsize.x || j == grid.bsize.y || k == grid.bsize.z) {
                    out[cell_id] = 0.0;
                } else {
                    out[cell_id] = fabsf(out[cell_id] - data[cell_id]);
                }
            }
        }
    }
}

/* CXX API */

DeviceKernels::DeviceKernels(BlockGrid grid, int kernel_block_dim, int kernel_grid_dim)
    : grid(grid)
    , kernel_block_dim(kernel_block_dim)
    , kernel_grid_dim(kernel_grid_dim)
{
}

void DeviceKernels::get_border(
    double* out, double* data,
    int a_size, int b_size,
    int border_idx, layer_tag tag)
{
    START_KERNEL((get_border_kernel<<<BORDER_DIMS_2D>>>(
        out, data,
        grid, a_size, b_size,
        border_idx, tag)));
}

void DeviceKernels::set_border(
    double* dest, double* data,
    int a_size, int b_size,
    int border_idx, layer_tag tag)
{
    START_KERNEL((set_border_kernel<<<BORDER_DIMS_2D>>>(
        dest, data,
        grid, a_size, b_size,
        border_idx, tag)));
}

double DeviceKernels::compute(double* out, double* data, mydim3<double> height)
{
    START_KERNEL((
        compute_kernel<<<BORDER_DIMS_3D(kernel_block_dim, kernel_grid_dim)>>>(
            out, data, grid, height)));

    START_KERNEL((
        abs_error_kernel<<<BORDER_DIMS_3D(kernel_block_dim, kernel_grid_dim)>>>(
            data, out, grid)));

    thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(data);
    double error = *thrust::max_element(dev_ptr, dev_ptr + grid.cells_per_block());

    CUDA_ERR(cudaGetLastError());
    return error;
}

dim3_type side_tag_to_dim3_type(side_tag tag)
{
    switch (tag) {
    default:
    case LEFT:
    case RIGHT:
        return DIM3_TYPE_X;
    case TOP:
    case BOTTOM:
        return DIM3_TYPE_Z;
    case FRONT:
    case BACK:
        return DIM3_TYPE_Y;
    }
}

dim3_type layer_tag_to_dim3_type(layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return DIM3_TYPE_X;
    case VERTICAL:
        return DIM3_TYPE_Y;
    case FRONT_BACK:
        return DIM3_TYPE_Z;
    }
}

layer_tag dim3_type_to_layer_tag(dim3_type type)
{
    switch (type) {
    default:
    case DIM3_TYPE_X:
        return LEFT_RIGHT;
    case DIM3_TYPE_Y:
        return FRONT_BACK;
    case DIM3_TYPE_Z:
        return VERTICAL;
    }
}

layer_tag side_tag_to_layer_tag(side_tag tag)
{
    switch (tag) {
    default:
    case LEFT:
    case RIGHT:
        return LEFT_RIGHT;
    case TOP:
    case BOTTOM:
        return VERTICAL;
    case FRONT:
    case BACK:
        return FRONT_BACK;
    }
}
