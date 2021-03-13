#include "exchange.hpp"
#include "helpers.hpp"

Exchange::Exchange(const Grid& grid, const Task& task, Problem& problem)
    : grid(grid)
    , task(task)
    , problem(problem)
{
    // да, так будет оверхед по памяти, но не нужно мучиться с границами.
    const int max_dim = grid.max_size();

    this->send_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim));
    this->receive_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim));
}

enum {
    LEFT,
    RIGHT,
    FRONT,
    BACK,
    UP,
    DOWN
};

void Exchange::exchange2D(
    const int block_idx, const int block_size,
    const int a_size, const int b_size,
    const double lower_init, const double upper_init,
    const int recvtag_lower, const int recvtag_upper,
    std::function<size_t(int my, int a, int b)> get_cell_idx,
    std::function<int(int)> get_block_idx)
{
    MPI_Status status;

    int count = a_size * b_size;

    std::cout << "block_idx: " << block_idx << std::endl;

    if (block_idx == 0) {
        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < grid.bsize.z; ++b)
                problem.data[get_cell_idx(-1, a, b)] = lower_init;
    } else {
        // отсылка и прием нижнего граничного условия
        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < b_size; ++b)
                send_buffer[size_t(a * b_size + b)] = problem.data[get_cell_idx(0, a, b)];

        int exchange_process_rank = get_block_idx(block_idx - 1);

        std::cout << "process_rank:" << exchange_process_rank << std::endl;

        CSC(MPI_Sendrecv(send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_lower,
            receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_upper,
            MPI_COMM_WORLD, &status));

        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < grid.bsize.z; ++b)
                problem.data[get_cell_idx(-1, a, b)] = receive_buffer[size_t(a * b_size + b)];
    }

    std::cout << "block_size_top: " << block_size - 1 << std::endl;
    if (block_idx == block_size - 1) {
        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < b_size; ++b)
                problem.data[get_cell_idx(block_size, a, b)] = upper_init;
    } else {
        // отсылка и прием верхнего граничного условия
        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < b_size; ++b)
                send_buffer[size_t(a * grid.bsize.z + b)] = problem.data[get_cell_idx(block_idx - 1, a, b)];

        int exchange_process_rank = get_block_idx(block_idx + 1);
        std::cout << "process_rank:" << exchange_process_rank << std::endl;

        CSC(MPI_Sendrecv(send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_upper,
            receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_lower,
            MPI_COMM_WORLD, &status));

        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < b_size; ++b)
                problem.data[get_cell_idx(block_size, a, b)] = receive_buffer[size_t(a * b_size + b)];
    }
}

void Exchange::boundary_layer_exchange()
{
    const dim3<int> block_dim = grid.block_dim();
    std::cout << block_dim.print("block_dim") << std::endl;

    exchange2D(
        block_dim.x, grid.n_blocks.x,
        grid.bsize.y, grid.bsize.z,
        task.u_left, task.u_right,
        LEFT, RIGHT,
        [this](int my, int a, int b) { return grid.cell_idx(my, a, b); },
        [this, block_dim](int block_idx) {
            return grid.block_idx(block_idx, block_dim.y, block_dim.z);
        });

    exchange2D(
        block_dim.y, grid.n_blocks.y,
        grid.bsize.x, grid.bsize.z,
        task.u_front, task.u_back,
        FRONT, BACK,
        [this](int my, int a, int b) { return grid.cell_idx(a, my, b); },
        [this, block_dim](int block_idx) {
            return grid.block_idx(block_dim.x, block_idx, block_dim.z);
        });

    exchange2D(
        block_dim.z, grid.n_blocks.z,
        grid.bsize.x, grid.bsize.y,
        task.u_bottom, task.u_top,
        FRONT, BACK,
        [this](int my, int a, int b) { return grid.cell_idx(a, b, my); },
        [this, block_dim](int block_idx) {
            return grid.block_idx(block_dim.x, block_dim.y, block_idx);
        });
}

void Exchange::write_layer(int j, int k, int block_idx, std::ostream& out)
{
    MPI_Status status;

    if (block_idx == 0) {
        for (int i = 0; i < grid.bsize.x; ++i) {
            receive_buffer[size_t(i)] = problem.data[grid.cell_idx(i, j, k)];
        }
    } else {
        CSC(MPI_Recv(receive_buffer.data(), grid.bsize.x, MPI_DOUBLE, block_idx, k * grid.bsize.z + j, MPI_COMM_WORLD, &status));
    }

    for (int i = 0; i < grid.bsize.x; ++i) {
        if (i != 0) {
            out << " ";
        }
        out << receive_buffer[size_t(i)];
    }
}

void Exchange::write_result(std::ostream& out)
{
    out << std::scientific;

    for (int bk = 0; bk < grid.n_blocks.z; ++bk) {
        for (int k = 0; k < grid.bsize.z; ++k) {
            for (int bj = 0; bj < grid.n_blocks.y; ++bj) {
                for (int j = 0; j < grid.bsize.y; ++j) {
                    for (int bi = 0; bi < grid.n_blocks.x; ++bi) {
                        int block_idx = grid.block_idx(bi, bj, bk);
                        this->write_layer(j, k, block_idx, out);
                    }
                    out << std::endl;
                }
            }
            out << std::endl;
        }
    }
}
