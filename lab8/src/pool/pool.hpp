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
// возвращает является ли border нижней границей для указанной оси.
bool check_is_lower(layer_tag split_type, side_tag border);

class GPU_pool {
    const layer_tag split_type;
    const Grid& grid;
    const mydim3<double> height;
    Task task;

    struct Elem : public DeviceProblem {
        std::vector<double> host_data;

        double* gpu_data;
        double* gpu_data_next;

        int load_border(layer_tag tag, side_tag border);
        int store_border(layer_tag tag, side_tag border);

        double calculate(mydim3<double> height);

        Elem(const BlockGrid& grid, int max_dim);
        ~Elem();

    private:
        void swap();
    };

    std::vector<Elem> devices;

    void init_devices(int max_dim);
    int get_devices() const;

    void stacked_data(side_tag border, bool from_device);

public:
    std::vector<double> data;

    GPU_pool(const Grid& grid, Task task);

    // TODO texcl=true

    // загружает данные с GPU в поле data.
    void load_gpu_data(side_tag tag);
    // загружает данные на GPU из поля data.
    void store_gpu_data(side_tag tag);

    // выполняет вычисления
    double calc();
};

#endif
