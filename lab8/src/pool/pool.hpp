#ifndef POOL_HPP
#define POOL_HPP

#include <vector>

#include "grid/grid.hpp"
#include "kernels.hpp"

struct split_by {
    int part_size, rest, n_parts;
    split_by(int need_split, int n_parts, int min_part_size);
};

class GPU_pool {
    const Grid& grid;
    const layer_tag split_type;

    struct Elem {
        BlockGrid grid;

        int* gpu_data;
        int* gpu_data_next;
        int* gpu_buffer;

        Elem(const BlockGrid& grid, int max_dim);
        ~Elem();
    };

    std::vector<int> data;
    std::vector<int> data_next;
    std::vector<int> buffer;

    std::vector<Elem> devices;

    void init_devices(int max_dim);
    void free_devices();

public:
    GPU_pool(int n_devices, const Grid& grid);
    ~GPU_pool();
};

#endif
