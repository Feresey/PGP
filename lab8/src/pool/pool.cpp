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

Device::Device(const Grid& grid, Task task)
    : grid(grid)
    , height(grid.height(task.l_size))
    , task(task)
    , device(grid, 32, 8)
    , data(grid.cells_per_block(), task.u_0)
{
    device.store_data(data);
}

void Device::load_gpu_border(side_tag border) { device.load_border(data, border); }
void Device::store_gpu_border(side_tag border)
{
    // debug("store border: %d", border);
    device.store_border(data, border);
    // debug("after store");
    // show(std::cerr);
}
void Device::load_gpu_data() { device.load_data(data); }

void Device::show(std::ostream& out)
{
    std::vector<double> temp(data.capacity());
    device.load_data(temp);
    for (int k = -1; k <= grid.bsize.z; ++k) {
        for (int j = -1; j <= grid.bsize.y; ++j) {
            for (int i = -1; i <= grid.bsize.x; ++i) {
                out << temp[grid.cell_absolute_id(i, j, k)] << " ";
            }
            out << std::endl;
        }
        out << std::endl;
    }
    out << std::endl;
}

double Device::calc()
{
    // show(std::cerr);
    double err = device.calculate(height);
    // show(std::cerr);
    return err;
}
