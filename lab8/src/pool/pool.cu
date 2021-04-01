#include <numeric>

#include "kernels.hpp"
#include "pool.hpp"

#include <cassert>

Device::Elem::Elem(const BlockGrid& grid, int kernel_grid_dim, int kernel_block_dim)
    : grid(grid)
    , DeviceKernels(grid, kernel_grid_dim, kernel_block_dim)
{
    const uint data_size = uint(grid.cells_per_block()) * sizeof(double);
    debug("init gpu array with %d size", data_size);
    CUDA_ERR(cudaMalloc(&gpu_data, data_size));
    CUDA_ERR(cudaMalloc(&gpu_data_next, data_size));

    const int max_dim = *grid.bsize.max_dim();
    CUDA_ERR(cudaMalloc(&gpu_buffer, max_dim * max_dim * sizeof(double)));
}

Device::Elem::~Elem()
{
    CUDA_ERR(cudaFree(gpu_data));
    CUDA_ERR(cudaFree(gpu_data_next));
    CUDA_ERR(cudaFree(gpu_buffer));
}

int get_layer_idx(const BlockGrid& grid, side_tag border)
{
    switch (border) {
    default:
    case LEFT:
    case BOTTOM:
    case FRONT:
        return -1;
    case RIGHT:
        return grid.bsize.x;
    case TOP:
        return grid.bsize.z;
    case BACK:
        return grid.bsize.y;
    }
}

std::pair<int, int> other_sizes(const BlockGrid& grid, layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return { grid.bsize.z, grid.bsize.y };
    case VERTICAL:
        return { grid.bsize.y, grid.bsize.x };
    case FRONT_BACK:
        return { grid.bsize.z, grid.bsize.x };
    }
}

void Device::Elem::load_data(std::vector<double>& dst)
{
    const uint data_size = uint(grid.cells_per_block()) * sizeof(double);
    CUDA_ERR(cudaMemcpy(dst.data(), gpu_data, data_size, cudaMemcpyDeviceToHost));
}

void Device::Elem::store_data(std::vector<double>& src)
{
    const uint data_size = uint(grid.cells_per_block()) * sizeof(double);
    CUDA_ERR(cudaMemcpy(gpu_data, src.data(), data_size, cudaMemcpyHostToDevice));
}

void Device::Elem::load_border(std::vector<double>& dst, side_tag border)
{
    // debug("load border %d", border);
    int layer_idx = get_layer_idx(grid, border);
    if (layer_idx == -1) {
        ++layer_idx;
    } else {
        --layer_idx;
    }
    layer_tag layer = side_tag_to_layer_tag(border);
    std::pair<int, int> sizes = other_sizes(grid, layer);
    const int data_size = sizes.first * sizes.second;
    this->get_border(gpu_buffer, gpu_data, sizes.first, sizes.second, layer_idx, layer);
    CUDA_ERR(cudaMemcpy(dst.data(), gpu_buffer, data_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void Device::Elem::store_border(std::vector<double>& src, side_tag border)
{
    // debug("store border %d", border);
    int layer_idx = get_layer_idx(grid, border);
    layer_tag layer = side_tag_to_layer_tag(border);
    std::pair<int, int> sizes = other_sizes(grid, layer);
    const int data_size = sizes.first * sizes.second;
    CUDA_ERR(cudaMemcpy(gpu_buffer, src.data(), data_size * sizeof(double), cudaMemcpyHostToDevice));
    this->set_border(gpu_data, gpu_buffer, sizes.first, sizes.second, layer_idx, layer);
}

double Device::Elem::calculate(mydim3<double> height)
{
    // debug("compute");
    double err = this->compute(gpu_data_next, gpu_data, height);
    std::swap(gpu_data, gpu_data_next);
    return err;
}
