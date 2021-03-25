#include <numeric>

#include "kernels.hpp"
#include "pool.hpp"

#include <cassert>

GPU_pool::Elem::Elem(const BlockGrid& grid)
    : grid(grid)
    , DeviceProblem(grid)
{
    const uint buffer_size = uint(grid.cells_per_block()) * sizeof(double);
    debug("init gpu array with %d size", buffer_size);
    CUDA_ERR(cudaMalloc(&gpu_data, buffer_size));
    CUDA_ERR(cudaMalloc(&gpu_data_next, buffer_size));
}

GPU_pool::Elem::~Elem()
{
    debug("free array");
    CUDA_ERR(cudaFree(gpu_data));
    CUDA_ERR(cudaFree(gpu_data_next));
}

int get_layer_idx(const BlockGrid& grid, side_tag border)
{
    switch (border) {
    default:
    case LEFT:
    case TOP:
    case FRONT:
        return -1;
    case RIGHT:
        return grid.bsize.x;
    case BOTTOM:
        return grid.bsize.y;
    case BACK:
        return grid.bsize.z;
    }
}

std::pair<int, int> other_sizes(const BlockGrid& grid, layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return { grid.bsize.y, grid.bsize.z };
    case VERTICAL:
        return { grid.bsize.x, grid.bsize.z };
    case FRONT_BACK:
        return { grid.bsize.x, grid.bsize.y };
    }
}

void GPU_pool::Elem::load_data(std::vector<double>& dst)
{
    const uint buffer_size = uint(grid.cells_per_block()) * sizeof(double);
    CUDA_ERR(cudaMemcpy(dst.data(), gpu_data, buffer_size, cudaMemcpyDeviceToHost));
}

void GPU_pool::Elem::store_data(std::vector<double>& src)
{
    const uint buffer_size = uint(grid.cells_per_block()) * sizeof(double);
    CUDA_ERR(cudaMemcpy(gpu_data, src.data(), buffer_size, cudaMemcpyHostToDevice));
}

void GPU_pool::Elem::load_border(std::vector<double>& dst, side_tag border)
{
    // debug("load border %d", border);
    int layer_idx = get_layer_idx(grid, border);
    layer_tag layer = side_tag_to_layer_tag(border);
    std::pair<int, int> sizes = other_sizes(grid, layer);
    const int data_size = sizes.first * sizes.second;
    this->get_border(gpu_data_next, gpu_data, sizes.first, sizes.second, layer_idx, layer);
    CUDA_ERR(cudaMemcpy(dst.data(), gpu_data_next, data_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_pool::Elem::store_border(std::vector<double>& src, side_tag border)
{
    // debug("store border %d", border);
    int layer_idx = get_layer_idx(grid, border);
    layer_tag layer = side_tag_to_layer_tag(border);
    std::pair<int, int> sizes = other_sizes(grid, layer);
    // debug("on device");
    // for (int i = 0; i < sizes.first; ++i) {
    //     for (int j = 0; j < sizes.second; ++j) {
    //         std::cerr << src[i * sizes.second + j] << " ";
    //     }
    //     std::cerr << std::endl;
    // }
    // std::cerr << std::endl;
    // debug("on device end");
    const int data_size = sizes.first * sizes.second;
    CUDA_ERR(cudaMemcpy(gpu_data_next, src.data(), data_size * sizeof(double), cudaMemcpyHostToDevice));
    this->set_border(gpu_data, gpu_data_next, sizes.first, sizes.second, layer_idx, layer);
}

double GPU_pool::Elem::calculate(mydim3<double> height)
{
    double err = this->compute(gpu_data_next, gpu_data, height);
    std::swap(gpu_data, gpu_data_next);
    return err;
}
