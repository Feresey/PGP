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

// Нет смысла делить маленькие данные, проще (и быстрее) запихнуть их на один GPU
#define MIN_GPU_SPLIT_SIZE 5000

GPU_pool::GPU_pool(const Grid& grid, Task task)
    : grid(grid)
    , task(task)
    //@ данные для разных GPU будут разделяться по наибольшей стороне блока.@
    //@ Да, можно сделать лучше.@
    , split_type(dim3_type_to_layer_tag(grid.bsize.max_dim().get_type()))
{
    int n_devices = this->get_devices();

    auto max_elem = grid.bsize.max_dim();
    int max_dim = *max_elem;

    //@ сторона делится на n_devices частей, остаток добавляется последнему блоку@
    const split_by& split = split_by(max_dim, n_devices, MIN_GPU_SPLIT_SIZE);

    mydim3<int> init_dim;
    mydim3<int> rest_dim;
    for (auto elem = grid.bsize.begin(); elem != grid.bsize.end(); ++elem) {
        auto offset = std::distance(grid.bsize.begin(), elem);
        if (elem == max_elem) {
            init_dim[offset] = split.part_size;
            rest_dim[offset] = split.part_size + split.rest;
        } else {
            init_dim[offset] = grid.bsize[offset];
            rest_dim[offset] = grid.bsize[offset];
        }
    }

    int max_rest_dim = *rest_dim.max_dim();

    this->devices = std::vector<Elem>(size_t(split.n_parts), Elem(BlockGrid { init_dim }, max_rest_dim));
    this->devices.back() = Elem(BlockGrid { rest_dim }, max_rest_dim);

    this->init_devices(max_dim);
}
