#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <vector>

#include "grid/grid.hpp"

struct Task {
    mydim3<double> l_size;
    double eps;
    double u_bottom, u_top, u_left, u_right, u_front, u_back;
    double u_0;

    friend std::istream& operator>>(std::istream& in, Task& data);
    friend std::ostream& operator<<(std::ostream& out, const Task& data);

    void mpi_bcast();
};

class Problem {
    const Grid& grid;
    const Task& task;

    mydim3<double> height;

    std::vector<double> data_next;

public:
    std::vector<double> data;
    double calc();

    Problem(const Task& task, const Grid& grid);

    void show(std::ostream& out) const;
};

#endif