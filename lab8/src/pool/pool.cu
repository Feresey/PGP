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

void GPU_pool::init_devices(int max_dim)
{
    const dim3_type type = layer_tag_to_dim3_type(split_type);
    std::vector<double> temp(max_dim * max_dim);

    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        Elem& device = devices[device_id];

        const uint buffer_size = uint(device.grid.cells_per_block()) * sizeof(double);

        device.set_device(device_id);

        CUDA_ERR(cudaMalloc(&device.gpu_data, buffer_size));
        CUDA_ERR(cudaMalloc(&device.gpu_data_next, buffer_size));

        for (int i = -1; i <= device.grid.bsize.x; ++i) {
            for (int j = -1; j <= device.grid.bsize.y; ++j) {
                for (int k = -1; k <= device.grid.bsize.z; ++k) {
                    const mydim3<int> cell = { i, j, k };

                    mydim3<int> abused_cell = cell;
                    abused_cell[type] += device.grid.bsize[type] * device_id;

                    temp[device.grid.cell_absolute_id(cell)] = buffer[device.grid.cell_absolute_id(abused_cell)];
                }
            }
        }

        CUDA_ERR(cudaMemcpy(device.gpu_data, temp.data(), buffer_size, cudaMemcpyHostToDevice));
    }
}
