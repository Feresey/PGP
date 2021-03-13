#include <cmath>

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

    std::ostream& out = std::cerr;

    const dim3<double> inv_h = {
        1.0 / (height.x * height.x),
        1.0 / (height.y * height.y),
        1.0 / (height.z * height.z)
    };

    out
        << height.print("height")
        << std::endl
        << inv_h.print("inv_h")
        << std::endl;

    // show(out);

    for (int i = 0; i < grid.bsize.x; ++i) {
        for (int j = 0; j < grid.bsize.y; ++j) {
            for (int k = 0; k < grid.bsize.z; ++k) {
                out << "idx: " << grid.cell_absolute_id(i - 1, j, k) << " ";
                double num = 0.0;

                num += (data[grid.cell_absolute_id(i + 1, j, k)] + data[grid.cell_absolute_id(i - 1, j, k)]) * inv_h.x;
                out << "num: (" << data[grid.cell_absolute_id(i + 1, j, k)] << "," << data[grid.cell_absolute_id(i - 1, j, k)] << ") ";
                num += (data[grid.cell_absolute_id(i, j + 1, k)] + data[grid.cell_absolute_id(i, j - 1, k)]) * inv_h.y;
                num += (data[grid.cell_absolute_id(i, j, k + 1)] + data[grid.cell_absolute_id(i, j, k - 1)]) * inv_h.z;

                double denum = 2.0 * (inv_h.x + inv_h.y + inv_h.z);
                double temp = num / denum;
                double error = std::fabs(data[grid.cell_absolute_id(i, j, k)] - temp);

                if (error > max_error) {
                    max_error = error;
                }

                data_next[grid.cell_absolute_id(i, j, k)] = temp;
            }
            out << std::endl;
        }
        out << std::endl;
    }

    std::swap(data, data_next);
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