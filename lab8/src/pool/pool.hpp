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

class GPU_pool {
    const Grid& grid;
    const layer_tag split_type;

    Task task;
    mydim3<double> height;

    std::vector<int> data_next;
    std::vector<int> buffer;

    struct Elem {
        BlockGrid grid;

        std::vector<double> host_data;
        std::vector<double> host_buffer;

        double* gpu_data;
        double* gpu_data_next;
        double* gpu_buffer;

        Elem(const BlockGrid& grid, int max_dim);
        ~Elem();
    };

    std::vector<Elem> devices;

    void init_devices(int max_dim);

    int get_devices()const;

public:
    std::vector<int> data;

    GPU_pool(const Grid& grid, Task task);

    double calc();
};

#endif
