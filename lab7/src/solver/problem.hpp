#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <vector>

#include "../grid.hpp"

class Problem {
    Grid grid;
    dim3<double> height;
    dim3<int> bsize;

    std::vector<double> data, data_next;

public:
    double calc();

    const std::vector<double>& get_result() const;

    Problem();
    Problem(size_t data_size, double init_value, const dim3<int> bsize, const dim3<double> height);
};

#endif