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

public:
    Exchange(const Grid& grid, const Task& task, Device& pool);
    void boundary_layer_exchange();

    void write_result(const std::string& output);
    void send_result();
};

#endif
