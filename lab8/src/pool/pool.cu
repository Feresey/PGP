#include "kernels.hpp"
#include "pool.hpp"

GPU_pool::Elem::Elem(const BlockGrid& grid, int max_dim)
    : DeviceProblem(grid)
    , host_data(max_dim * max_dim)
{
}

GPU_pool::Elem::~Elem()
{
    CUDA_ERR(cudaFree(gpu_data));
    CUDA_ERR(cudaFree(gpu_data_next));
    CUDA_ERR(cudaFree(gpu_buffer));
}

std::pair<int, int> other_sizes(const BlockGrid& grid, layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return { grid.bsize.y, grid.bsize.z };
    case FRONT_BACK:
        return { grid.bsize.x, grid.bsize.z };
    case VERTICAL:
        return { grid.bsize.x, grid.bsize.y };
    }
}

void GPU_pool::Elem::load_border(int layer_idx, layer_tag tag)
{
    std::pair<int, int> sizes = other_sizes(grid, tag);
    this->get_border(gpu_data_next, gpu_data, sizes.first, sizes.second, layer_idx, tag);
    CUDA_ERR(cudaMemcpy(host_data.data(), gpu_data_next, sizes.first * sizes.second * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_pool::Elem::store_border(int layer_idx, layer_tag tag)
{
    std::pair<int, int> sizes = other_sizes(grid, tag);
    CUDA_ERR(cudaMemcpy(gpu_data_next, host_data.data(), sizes.first * sizes.second * sizeof(double), cudaMemcpyHostToDevice));
    this->set_border(gpu_data, gpu_data_next, sizes.first, sizes.second, layer_idx, tag);
}

int GPU_pool::get_devices() const
{
    int n_devices;
    CUDA_ERR(cudaGetDeviceCount(&n_devices));
    return n_devices;
}

void GPU_pool::init_devices(int max_dim)
{
    std::vector<double> temp(max_dim * max_dim);

    for (auto device_it = devices.begin(); device_it != devices.end(); ++device_it) {
        Elem& device = *device_it;
        int device_id = int(std::distance(devices.begin(), device_it));

        const uint buffer_size = uint(device.grid.cells_per_block()) * sizeof(double);

        device.set_device(device_id);

        CUDA_ERR(cudaMalloc(&device.gpu_data, buffer_size));
        CUDA_ERR(cudaMalloc(&device.gpu_data_next, buffer_size));
        CUDA_ERR(cudaMalloc(&device.gpu_buffer, temp.size() * sizeof(double)));

        for (int k = -1; k <= device.grid.bsize.z; ++k) {
            for (int j = -1; j <= device.grid.bsize.y; ++j) {
                for (int i = -1; i <= device.grid.bsize.x; ++i) {
                    const mydim3<int> cell = { i, j, k };
                    const dim3_type type = layer_tag_to_dim3_type(split_type);

                    mydim3<int> abused_cell = cell;
                    abused_cell[type] = cell[type] + device.grid.bsize[type] * device_id;

                    temp[device.grid.cell_absolute_id(cell)] = buffer[device.grid.cell_absolute_id(abused_cell)];
                }
            }
        }

        CUDA_ERR(cudaMemcpy(device.gpu_data, temp.data(), buffer_size, cudaMemcpyHostToDevice));
    }
}
