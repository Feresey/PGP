#include <cmath>

#include "problem.hpp"

Problem::Problem() { }

Problem::Problem(size_t data_size, double init_value, const dim3<int> bsize, const dim3<double> height)
    : height(height)
    , bsize(bsize)
    , data(data_size, init_value)
    , data_next(data_size)
{
}

double Problem::calc()
{
    double max_error = 0.0;
    for (int i = 0; i < grid.bsize.x; ++i)
        for (int j = 0; j < grid.bsize.y; ++j)
            for (int k = 0; k < grid.bsize.z; ++k) {
                double inv_hxsqr = 1.0 / (height.x * height.x);
                double inv_hysqr = 1.0 / (height.y * height.y);
                double inv_hzsqr = 1.0 / (height.z * height.z);

                double num = (data[grid.cell_idx(i + 1, j, k)] + data[grid.cell_idx(i - 1, j, k)]) * inv_hxsqr
                    + (data[grid.cell_idx(i, j + 1, k)] + data[grid.cell_idx(i, j - 1, k)]) * inv_hysqr
                    + (data[grid.cell_idx(i, j, k + 1)] + data[grid.cell_idx(i, j, k - 1)]) * inv_hzsqr;
                double denum = 2.0 * (inv_hxsqr + inv_hysqr + inv_hzsqr);
                double temp = num / denum;
                double error = std::abs(data[grid.cell_idx(i, j, k)] - temp);

                if (error > max_error)
                    max_error = error;

                data_next[grid.cell_idx(i, j, k)] = temp;
            }

    std::swap(data, data_next);
    return max_error;
}
