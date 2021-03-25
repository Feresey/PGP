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

    std::vector<double> buffer;

    struct Elem : public DeviceProblem {
        std::vector<double> host_data;

        double* gpu_data;
        double* gpu_data_next;

        int load_border(layer_tag tag, side_tag border);
        int load_data();
        int store_border(layer_tag tag, side_tag border);
        int store_data();

        double calculate(mydim3<double> height);

        Elem(const BlockGrid& grid);
        Elem(Elem&& elem);
        ~Elem();
    };

    std::vector<Elem> devices;

    int get_devices() const;
    // соединяет и разделяет данные с нескольких GPU
    void stacked_borders(side_tag border, bool from_device);
    // void stacked_border(side_tag border)
    void move_gpu_data(bool to_device);

public:
    std::vector<double> data;

    GPU_pool(const Grid& grid, Task task);

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
