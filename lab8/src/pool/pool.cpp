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
        if (n_parts_normalized == 1) {
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
{
    int n_devices = this->get_devices();

    auto max_elem = grid.bsize.max_dim();
    int max_dim = *max_elem;

    // сторона делится на n_devices частей, остаток добавляется последнему блоку
    const split_by& split = split_by(max_dim, n_devices, MIN_GPU_SPLIT_SIZE);

    mydim3<int> init_dim;
    mydim3<int> rest_dim;
    // итератор по координатам x, y, z
    for (auto elem = grid.bsize.begin(); elem != grid.bsize.end(); ++elem) {
        dim3_type offset = elem.get_type();
        // координата, по которой разделение
        if (elem == max_elem) {
            init_dim[offset] = split.part_size;
            rest_dim[offset] = split.part_size + split.rest;
        } else {
            init_dim[offset] = *elem;
            rest_dim[offset] = *elem;
        }
    }

    int max_rest_dim = *rest_dim.max_dim();

    this->devices = std::vector<Elem>(size_t(split.n_parts), Elem(BlockGrid { init_dim }, max_rest_dim));
    this->devices.back() = Elem(BlockGrid { rest_dim }, max_rest_dim);

    this->init_devices(max_dim);
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

void GPU_pool::stacked_data(side_tag border, bool from_device)
{
    auto res_dims = other_sizes(grid, split_type);
    stack_t stack_type = get_stack_type(split_type, border);

    int offset_a = 0;
    int offset_b = 0;

    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        Elem& device = devices[device_id];
        std::pair<int, int> sizes = other_sizes(device.grid, split_type);
        const int data_size = sizes.first * sizes.second;

        std::vector<double>& src = (from_device ? device.host_data : data);
        std::vector<double>& dst = (from_device ? data : device.host_data);

        switch (stack_type) {
        case STACK_HORIZONTAL:
            offset_a = offset_b;
            // построчное копирование
            for (int read_a = 0; read_a < sizes.first; ++read_a) {
                std::copy(
                    src.begin() + offset_a,
                    src.begin() + offset_a + sizes.second,
                    dst.begin());
                offset_a += res_dims.first;
            }
            offset_b += sizes.second;
            break;
        case STACK_VERTICAL:
            // повезло, повезло
            std::copy(
                src.begin() + offset_a,
                src.begin() + offset_a + data_size,
                dst.begin());
            offset_a += data_size;
            break;
        }
    }
}

void GPU_pool::load_gpu_data(side_tag border)
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

    this->stacked_data(border, true);
}

void GPU_pool::store_gpu_data(side_tag border)
{
    // особый случай. Нужная граница принадлежит только одной GPU
    if ((border & split_type) != 0) {
        bool is_lower = check_is_lower(split_type, border);
        const size_t one_shot_idx = is_lower ? 0UL : (devices.size() - 1UL);

        Elem& device = devices[one_shot_idx];
        std::pair<int, int> sizes = other_sizes(grid, split_type);
        int data_size = sizes.first * sizes.second;

        std::copy(data.begin(), data.begin() + data_size, device.host_data.begin());
        device.set_device(int(one_shot_idx));
        device.store_border(split_type, border);

        return;
    }

    // Все остальные случаи. Нужная граница находится на нескольких GPU.
    for (size_t device_id = 0; device_id < devices.size(); ++device_id) {
        Elem& device = devices[device_id];
        device.set_device(int(device_id));
        device.store_border(split_type, border);
    }

    this->stacked_data(border, false);
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