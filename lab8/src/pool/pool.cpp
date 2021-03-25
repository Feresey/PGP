#include <algorithm>
#include <numeric>

#include "kernels.hpp"
#include "pool.hpp"

// split_by возвращает std::pair<part_size, rest>
split_by::split_by(int need_split, int n_parts, int min_part_size)
{
    part_size = need_split / n_parts;
    if (part_size < min_part_size) {
        int n_parts_normalized = need_split / min_part_size;
        // нет смысла делить на 1 целую часть и ещё 1 часть с "хвостиком"
        if (n_parts_normalized <= 1) {
            this->part_size = need_split;
            this->rest = 0;
            this->n_parts = 1;
            return;
        }
        this->part_size = need_split / n_parts_normalized;
        this->rest = need_split % n_parts_normalized;
        this->n_parts = n_parts_normalized;
        return;
    }
    this->rest = need_split % n_parts;
    this->n_parts = n_parts;
}

// Нет смысла делить маленькие данные. Проще (и быстрее) запихнуть их на один GPU
#define MIN_GPU_SPLIT_SIZE 5000

GPU_pool::GPU_pool(const Grid& grid, Task task)
    // данные для разных GPU будут разделяться по наибольшей стороне блока.
    // Да, можно сделать лучше.
    : split_type(dim3_type_to_layer_tag(grid.bsize.max_dim().get_type()))
    , grid(grid)
    , task(task)
    , buffer(grid.cells_per_block())
    , data(grid.cells_per_block(), task.u_0)
{
    debug("cells per block: %ld", grid.cells_per_block());
    std::cerr << grid << std::endl;
    int n_devices = this->get_devices();

    auto max_elem = grid.bsize.max_dim();
    int max_dim = *max_elem;

    // сторона делится на n_devices частей, остаток добавляется последнему блоку
    const split_by& split = split_by(max_dim, n_devices, MIN_GPU_SPLIT_SIZE);

    mydim3<int> init_dim;
    mydim3<int> rest_dim;
    // итератор по координатам x, y, z
    for (auto elem = grid.bsize.begin(); elem != grid.bsize.end(); ++elem) {
        // координата, по которой разделение
        dim3_type type = elem.get_type();
        if (elem == max_elem) {
            init_dim[type] = split.part_size;
            rest_dim[type] = split.part_size + split.rest;
        } else {
            init_dim[type] = *elem;
            rest_dim[type] = *elem;
        }
    }

    debug("devices head");
    for (size_t device_id = 0; device_id < split.n_parts; ++device_id) {
        devices.push_back(Elem(BlockGrid { init_dim }));
        debug("fuck");
    }
    if (split.rest != 0) {
        debug("devices tail");
        this->devices.push_back(Elem(BlockGrid { rest_dim }));
    }

    debug("before init");
    this->move_gpu_data(true);
    debug("after init");
}

std::pair<int, int> other_sizes(const BlockGrid& grid, layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return { grid.bsize.y, grid.bsize.z };
    case VERTICAL:
        return { grid.bsize.x, grid.bsize.z };
    case FRONT_BACK:
        return { grid.bsize.x, grid.bsize.y };
    }
}

// возвращает является ли border нижней границей для указанной оси.
bool check_is_lower(layer_tag split_type, side_tag border)
{
    switch (split_type) {
    case LEFT_RIGHT:
        return (border == LEFT);
    case VERTICAL:
        return (border == BOTTOM);
    case FRONT_BACK:
        return (border == FRONT);
    default:
        return -1;
    }
}

enum stack_t {
    STACK_VERTICAL,
    STACK_HORIZONTAL
};

stack_t get_stack_type(layer_tag split_type, side_tag border)
{
    switch (split_type) {
    default:
    case LEFT_RIGHT:
        return STACK_HORIZONTAL;
    case FRONT_BACK:
        return STACK_VERTICAL;
    case VERTICAL:
        switch (border) {
        case TOP:
        case BOTTOM:
            return STACK_HORIZONTAL;
        default:
            return STACK_VERTICAL;
        }
    }
}

void GPU_pool::stacked_borders(side_tag border, bool from_device)
{
    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        Elem& device = devices[device_id];
        device.set_device(int(device_id));
        
    }
}

void GPU_pool::load_gpu_border(side_tag border)
{
    // особый случай. Нужная граница принадлежит только одной GPU
    if ((border & split_type) != 0) {
        bool is_lower = check_is_lower(split_type, border);

        const size_t one_shot_idx = is_lower ? 0UL : (devices.size() - 1UL);

        Elem& device = devices[one_shot_idx];
        device.set_device(int(one_shot_idx));
        int data_size = device.load_border(split_type, border);

        std::copy(device.host_data.begin(), device.host_data.begin() + data_size, data.begin());

        return;
    }

    // Все остальные случаи. Нужная граница находится на нескольких GPU.
    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        Elem& device = devices[device_id];
        device.set_device(int(device_id));
        device.load_border(split_type, border);
    }

    this->stacked_borders(border, true);
}

void GPU_pool::store_gpu_border(side_tag border)
{
    debug("store border %d", border);
    // особый случай. Нужная граница принадлежит только одной GPU
    if ((border & split_type) != 0) {
        bool is_lower = check_is_lower(split_type, border);
        const size_t one_shot_idx = is_lower ? 0UL : (devices.size() - 1UL);

        Elem& device = devices[one_shot_idx];
        std::pair<int, int> sizes = other_sizes(grid, split_type);
        int data_size = sizes.first * sizes.second;

        debug("on host");
        for (int i = 0; i < sizes.first; ++i) {
            for (int j = 0; j < sizes.second; ++j) {
                std::cerr << data[i * sizes.second + j] << " ";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
        debug("on host end");

        std::copy(data.begin(), data.begin() + data_size, device.host_data.begin());
        device.set_device(int(one_shot_idx));
        device.store_border(split_type, border);

        return;
    }

    debug("stack data");
    this->stacked_borders(border, false);

    // Все остальные случаи. Нужная граница находится на нескольких GPU.
    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        Elem& device = devices[device_id];
        device.set_device(int(device_id));
        device.store_border(split_type, border);
    }
}

double GPU_pool::calc()
{
    double all_error = 0.0;
    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        auto& device = devices[device_id];
        device.set_device(int(device_id));
        double local_err = device.calculate(height);
        if (local_err > all_error) {
            all_error = local_err;
        }
    }
    return all_error;
}

void GPU_pool::load_gpu_data() { this->move_gpu_data(false); }
