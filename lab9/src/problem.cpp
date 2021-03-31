#include <cmath>
#include <omp.h>

#include "problem.hpp"

Problem::Problem(const Task& task, const Grid& grid)
    : grid(grid)
    , task(task)
    , height(grid.height(task.l_size))
    , data_next(grid.cells_per_block())
    , data(grid.cells_per_block(), task.u_0)
{
}

double Problem::calc()
{
    double max_error = 0.0;

    const dim3<double> inv_h = {
        1.0 / (height.x * height.x),
        1.0 / (height.y * height.y),
        1.0 / (height.z * height.z)
    };

    auto& data = this->data;

    int n_threads = omp_get_max_threads();
    for (int x = 0; x < grid.bsize.x; x += n_threads) {
        for (int y = 0; y < grid.bsize.y; y += n_threads) {
            for (int z = 0; z < grid.bsize.z; z += n_threads) {
                double error = 0.0;

#pragma omp parallel num_threads(n_threads) shared(data) reduction(max:max_error)
                {
                    int rank = omp_get_thread_num();

                    for (int bx = 0; bx < n_threads; ++bx) {
                        for (int by = 0; by < n_threads; ++by) {
                            int i = x + bx, j = y + by, k = z + rank;
                            if (i >= grid.bsize.x || j >= grid.bsize.y || k >= grid.bsize.z) {
                                continue;
                            }
                            double num = 0.0
                                + (data[grid.cell_absolute_id(i + 1, j, k)] + data[grid.cell_absolute_id(i - 1, j, k)]) * inv_h.x
                                + (data[grid.cell_absolute_id(i, j + 1, k)] + data[grid.cell_absolute_id(i, j - 1, k)]) * inv_h.y
                                + (data[grid.cell_absolute_id(i, j, k + 1)] + data[grid.cell_absolute_id(i, j, k - 1)]) * inv_h.z;

                            double denum = 2.0 * (inv_h.x + inv_h.y + inv_h.z);
                            double temp = num / denum;
                            error = std::fabs(data[grid.cell_absolute_id(i, j, k)] - temp);

                            if (error > max_error) {
                                max_error = error;
                            }

                            data_next[grid.cell_absolute_id(i, j, k)] = temp;
                        }
                    }
                }
            }
        }
    }

    std::swap(this->data, this->data_next);
    return max_error;
}

void Problem::show(std::ostream& out) const
{
    for (int k = -1; k <= grid.bsize.z; ++k) {
        for (int j = -1; j <= grid.bsize.y; ++j) {
            for (int i = -1; i <= grid.bsize.x; ++i) {
                if (i != -1) {
                    out << " ";
                }
                out << data[grid.cell_absolute_id(i, j, k)];
            }
            out << std::endl;
        }
        out << std::endl;
    }
    out << std::endl;
}
