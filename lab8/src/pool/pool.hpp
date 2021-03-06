#ifndef POOL_HPP
#define POOL_HPP

#include <vector>

#include "grid/grid.hpp"
#include "kernels.hpp"
#include "task.hpp"

struct split_by {
    int part_size, rest, n_parts;
    split_by(int need_split, int n_parts, int min_part_size);
};

std::pair<int, int> other_sizes(const BlockGrid& grid, layer_tag tag);

class Device {
    const Grid& grid;
    const mydim3<double> height;
    Task task;

    class Elem : DeviceKernels {
        const BlockGrid& grid;

        double* gpu_buffer;
        double* gpu_data;
        double* gpu_data_next;

    public:
        void load_border(std::vector<double>& dst, side_tag border);
        void load_data(std::vector<double>& dst);
        void store_border(std::vector<double>& src, side_tag border);
        void store_data(std::vector<double>& src);

        double calculate(mydim3<double> height);

        Elem(const BlockGrid& grid, int kernel_grid_dim, int kernel_block_dim);
        ~Elem();
    };

    Elem device;

    void move_border(side_tag border, bool to_device);

public:
    void show(std::ostream& out);
    std::vector<double> data;

    Device(const Grid& grid, Task task);

    // TODO texcl=true

    // загружает данные с GPU в поле data.
    void load_gpu_border(side_tag border);
    // загружает данные на GPU из поля data.
    void store_gpu_border(side_tag border);

    // выгружает все данные с GPU.
    void load_gpu_data();

    // выполняет вычисления
    double calc();
};

#endif
