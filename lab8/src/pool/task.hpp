#ifndef TASK_HPP
#define TASK_HPP

#include <iostream>

#include "dim3/dim3.hpp"

struct Task {
    mydim3<double> l_size;
    double eps;
    double u_bottom, u_top, u_left, u_right, u_front, u_back;
    double u_0;

    friend std::istream& operator>>(std::istream& in, Task& data);
    friend std::ostream& operator<<(std::ostream& out, const Task& data);

    void mpi_bcast();
};

#endif
