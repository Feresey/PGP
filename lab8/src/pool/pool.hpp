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
    const layer_tag split_type;
    const Grid& grid;
    const mydim3<double> height;
    Task task;

    std::vector<double> buffer;

    struct Elem : public DeviceProblem {
        std::vector<double> host_data;

        double* gpu_data;
        double* gpu_data_next;
        double* gpu_buffer;

        void load_border(int layer_idx, layer_tag tag);
        void store_border(int layer_idx, layer_tag tag);
        void set_device(int device_id) const;

        double compute(mydim3<double> height);

        Elem(const BlockGrid& grid, int max_dim);
        ~Elem();
    };

    std::vector<Elem> devices;

    void init_devices(int max_dim);
    int get_devices() const;

public:
    std::vector<double> data;

    GPU_pool(const Grid& grid, Task task);

    // texcl=true

    // загружает данные с GPU в поле data.
    void load_gpu_data(side_tag tag);
    // загружает данные на GPU из поля data.
    void store_gpu_data(side_tag tag);

    // выполняет вычисления
    double calc();
};

#endif
