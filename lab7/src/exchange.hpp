#ifndef EXCHANGE_HPP
#define EXCHANGE_HPP

#include <functional>

#include "grid/grid.hpp"
#include "problem.hpp"

class Exchange {
    Grid grid;
    Task task;
    Problem problem;

    std::vector<double> send_buffer;
    std::vector<double> receive_buffer;

    void exchange2D(
        const int block_idx, const int block_size,
        const int a_size, const int b_size,
        const double lower_init, const double upper_init,
        const int recvtag_lower, const int recvtag_upper,
        std::function<size_t(int my, int a, int b)> get_cell_idx,
        std::function<int(int)> get_block_idx);

public:
    Exchange(const Grid& grid, const Task& task, Problem& problem);
    void boundary_layer_exchange();
};

#endif
