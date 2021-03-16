#include "pool.hpp"

GPU_pool::Elem::Elem(const BlockGrid& grid, int max_dim)
    : grid(grid)
    , host_buffer(max_dim * max_dim)
{
}

GPU_pool::Elem::~Elem()
{
    CUDA_ERR(cudaFree(gpu_data));
    CUDA_ERR(cudaFree(gpu_data_next));
    CUDA_ERR(cudaFree(gpu_buffer));
}

int GPU_pool::get_devices() const {
    int n_devices;
	CUDA_ERR(cudaGetDeviceCount(&n_devices));
}

void GPU_pool::init_devices(int max_dim)
{
    std::vector<double> temp(max_dim * max_dim);

    for (auto device_it = devices.begin(); device_it != devices.end(); ++device_it) {
        Elem& device = *device_it;
        int device_id = int(std::distance(devices.begin(), device_it));

        const uint buffer_size = uint(device.grid.cells_per_block()) * sizeof(double);

        CUDA_ERR(cudaSetDevice(device_id));

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
