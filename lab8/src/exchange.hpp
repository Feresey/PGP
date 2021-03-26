#ifndef EXCHANGE_HPP
#define EXCHANGE_HPP

#include <functional>

#include "grid/grid.hpp"
#include "pool/pool.hpp"

class Exchange {
    const Grid& grid;
    const Task& task;

    Device& pool;

    std::vector<double> send_buffer;
    std::vector<double> receive_buffer;

    void exchange2D(dim3_type block_coord);

    void write_layer(int j, int k, int block_idx, std::ostream& out);

public:
    Exchange(const Grid& grid, const Task& task, Device& pool);
    void boundary_layer_exchange();

    void write_result(std::ostream& out);
    void send_result();
};

#endif
