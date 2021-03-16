#ifndef EXCHANGE_HPP
#define EXCHANGE_HPP

#include <functional>

#include "grid/grid.hpp"
#include "pool/pool.hpp"

class Exchange {
    const Grid& grid;
    const Task& task;

    GPU_pool& pool;

    std::vector<double> send_buffer;
    std::vector<double> receive_buffer;

    void exchange2D(
        const int block_idx, const int n_blocks, const int cell_size,
        const int a_size, const int b_size,
        const double lower_init, const double upper_init,
        const int recvtag_lower, const int recvtag_upper,
        std::function<size_t(int my, int a, int b)> get_cell_idx,
        std::function<int(int)> get_block_idx);

    void write_layer(int j, int k, int block_idx, std::ostream& out);

public:
    Exchange(const Grid& grid, const Task& task, GPU_pool& pool);
    void boundary_layer_exchange();

    void write_result(std::ostream& out);
    void send_result();
};

#endif
