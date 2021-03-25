#include <numeric>

#include "kernels.hpp"
#include "pool.hpp"

#include <cassert>

GPU_pool::Elem::Elem(const BlockGrid& grid)
    : DeviceProblem(grid)
{
    const uint buffer_size = uint(grid.cells_per_block()) * sizeof(double);
    host_data.resize(grid.cells_per_block());
    debug("init gpu array with %d size", buffer_size);
    CUDA_ERR(cudaMalloc(&gpu_data, buffer_size));
    CUDA_ERR(cudaMalloc(&gpu_data_next, buffer_size));
}

GPU_pool::Elem::Elem(Elem&& elem)
    : DeviceProblem(elem.grid)
{
    this->host_data = elem.host_data;
    this->gpu_data = elem.gpu_data;
    this->gpu_data_next = elem.gpu_data_next;
    elem.gpu_data = NULL;
    elem.gpu_data_next = NULL;
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
        return 0;
    case RIGHT:
        return grid.bsize.x;
    case BOTTOM:
        return grid.bsize.y;
    case BACK:
        return grid.bsize.z;
    }
}

int GPU_pool::Elem::load_data()
{
    const int data_size = grid.cells_per_block();
    CUDA_ERR(cudaMemcpy(host_data.data(), gpu_data, data_size * sizeof(double), cudaMemcpyDeviceToHost));
    return data_size;
}

int GPU_pool::Elem::load_border(layer_tag split_type, side_tag border)
{
    int layer_idx = get_layer_idx(grid, border);
    std::pair<int, int> sizes = other_sizes(grid, split_type);
    const int data_size = sizes.first * sizes.second;
    this->get_border(gpu_data_next, gpu_data, sizes.first, sizes.second, layer_idx, split_type);
    CUDA_ERR(cudaMemcpy(host_data.data(), gpu_data_next, data_size * sizeof(double), cudaMemcpyDeviceToHost));
    return data_size;
}

int GPU_pool::Elem::store_border(layer_tag tag, side_tag border)
{
    int layer_idx = get_layer_idx(grid, border);
    std::pair<int, int> sizes = other_sizes(grid, tag);
    debug("on device");
    for (int i = 0; i < sizes.first; ++i) {
        for (int j = 0; j < sizes.second; ++j) {
            std::cerr << host_data[i * sizes.second + j] << " ";
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
    debug("on device end");
    const int data_size = sizes.first * sizes.second;
    CUDA_ERR(cudaMemcpy(gpu_data_next, host_data.data(), data_size * sizeof(double), cudaMemcpyHostToDevice));
    this->set_border(gpu_data, gpu_data_next, sizes.first, sizes.second, layer_idx, tag);
    return data_size;
}

double GPU_pool::Elem::calculate(mydim3<double> height)
{
    this->compute(gpu_data_next, gpu_data, height);
    std::swap(gpu_data, gpu_data_next);
    return this->calc_abs_error(gpu_data_next, gpu_data);
}

int GPU_pool::get_devices() const
{
    int n_devices;
    CUDA_ERR(cudaGetDeviceCount(&n_devices));
    return n_devices;
}

void GPU_pool::move_gpu_data(bool to_device)
{
    // debug("move gpu data. data_size = %ld, to_device %d", data.size(), to_device);
    const dim3_type split_coord = layer_tag_to_dim3_type(split_type);

    int offset = 0;
    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        // debug("set device %ld", device_id);
        Elem& device = devices[device_id];

        if (!to_device) {
            device.set_device(int(device_id));
            device.load_data();
        }

        for (int i = -1; i <= device.grid.bsize.x; ++i) {
            for (int j = -1; j <= device.grid.bsize.y; ++j) {
                for (int k = -1; k <= device.grid.bsize.z; ++k) {
                    const mydim3<int> cell = { i, j, k };

                    mydim3<int> abused_cell = cell;
                    abused_cell[split_coord] += offset;

                    if (to_device) {
                        // debug("set point at %ld", grid.cell_absolute_id(abused_cell));
                        buffer[device.grid.cell_absolute_id(cell)] = data[grid.cell_absolute_id(abused_cell)];
                    } else {
                        // debug("load point at %ld", device.grid.cell_absolute_id(cell));
                        data[grid.cell_absolute_id(abused_cell)] = device.host_data[device.grid.cell_absolute_id(cell)];
                    }
                }
            }
        }
        offset += device.grid.bsize[split_coord];

        if (to_device) {
            const uint buffer_size = uint(device.grid.cells_per_block()) * sizeof(double);
            CUDA_ERR(cudaMemcpy(device.gpu_data, buffer.data(), buffer_size, cudaMemcpyHostToDevice));
        }
    }
}
