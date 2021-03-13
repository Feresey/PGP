#include "exchange.hpp"
#include "helpers.hpp"

Exchange::Exchange(const Grid& grid, const Task& task, Problem& problem)
    : grid(grid)
    , task(task)
    , problem(problem)
{
    // да, так будет оверхед по памяти, но не нужно мучиться с границами.
    const int max_dim = grid.max_size();

    this->send_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim + 2));
    this->receive_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim + 2));
}

enum {
    LEFT,
    RIGHT,
    FRONT,
    BACK,
    TOP,
    BOTTOM
};

void Exchange::exchange2D(
    const int block_idx, const int n_blocks, const int cell_size,
    const int a_size, const int b_size,
    const double lower_init, const double upper_init,
    const int recvtag_lower, const int recvtag_upper,
    std::function<size_t(int my, int a, int b)> get_cell_idx,
    std::function<int(int)> get_block_idx)
{

    MPI_Status status;

    int count = a_size * b_size;

    if (block_idx == 0) {
        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                problem.data[get_cell_idx(-1, a, b)] = lower_init;
            }
        }
    } else {
        // отсылка и прием нижнего граничного условия
        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                send_buffer[size_t(a * b_size + b)] = problem.data[get_cell_idx(0, a, b)];
            }
        }

        int exchange_process_rank = get_block_idx(block_idx - 1);

        CSC(MPI_Sendrecv(
            send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_lower,
            receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_upper,
            MPI_COMM_WORLD, &status));

        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                problem.data[get_cell_idx(-1, a, b)] = receive_buffer[size_t(a * b_size + b)];
            }
        }
    }

    if (block_idx == n_blocks - 1) {
        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                problem.data[get_cell_idx(cell_size, a, b)] = upper_init;
            }
        }
    } else {
        // отсылка и прием верхнего граничного условия
        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                send_buffer[size_t(a * b_size + b)] = problem.data[get_cell_idx(cell_size - 1, a, b)];
            }
        }

        int exchange_process_rank = get_block_idx(block_idx + 1);

        CSC(MPI_Sendrecv(
            send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_upper,
            receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, recvtag_lower,
            MPI_COMM_WORLD, &status));

        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                problem.data[get_cell_idx(cell_size, a, b)] = receive_buffer[size_t(a * b_size + b)];
            }
        }
    }
}

void Exchange::boundary_layer_exchange()
{
    const dim3<int> block_idx = grid.block_idx();

    exchange2D(
        block_idx.x, grid.n_blocks.x, grid.bsize.x,
        grid.bsize.y, grid.bsize.z,
        task.u_left, task.u_right,
        LEFT, RIGHT,
        [this](int my, int a, int b) { return grid.cell_absolute_id(my, a, b); },
        [this, block_idx](int my) {
            return grid.block_absolute_id(my, block_idx.y, block_idx.z);
        });

    exchange2D(
        block_idx.y, grid.n_blocks.y, grid.bsize.y,
        grid.bsize.x, grid.bsize.z,
        task.u_front, task.u_back,
        FRONT, BACK,
        [this](int my, int a, int b) { return grid.cell_absolute_id(a, my, b); },
        [this, block_idx](int my) {
            return grid.block_absolute_id(block_idx.x, my, block_idx.z);
        });

    exchange2D(
        block_idx.z, grid.n_blocks.z, grid.bsize.z,
        grid.bsize.x, grid.bsize.y,
        task.u_bottom, task.u_top,
        BOTTOM, TOP,
        [this](int my, int a, int b) { return grid.cell_absolute_id(a, b, my); },
        [this, block_idx](int my) {
            return grid.block_absolute_id(block_idx.x, block_idx.y, my);
        });
}

void Exchange::write_layer(int j, int k, int block_idx, std::ostream& out)
{
    MPI_Status status;

    if (block_idx == 0) {
        for (int i = 0; i < grid.bsize.x; ++i) {
            receive_buffer[size_t(i)] = problem.data[grid.cell_absolute_id(i, j, k)];
        }
    } else {
        int tag = k * grid.bsize.z + j;
        CSC(MPI_Recv(receive_buffer.data(), grid.bsize.x, MPI_DOUBLE, block_idx, tag, MPI_COMM_WORLD, &status));
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
                        int block_idx = grid.block_absolute_id(bi, bj, bk);
                        this->write_layer(j, k, block_idx, out);
                    }
                    out << std::endl;
                }
            }
            out << std::endl;
        }
    }
}

void Exchange::send_result()
{
    for (int k = 0; k < grid.bsize.z; ++k) {
        for (int j = 0; j < grid.bsize.y; ++j) {
            for (int i = 0; i < grid.bsize.x; ++i) {
                send_buffer[size_t(i)] = problem.data[grid.cell_absolute_id(i, j, k)];
            }
            int tag = k * grid.bsize.z + j;
            MPI_Send(send_buffer.data(), grid.bsize.x, MPI_DOUBLE, ROOT_RANK, tag, MPI_COMM_WORLD);
        }
    }
}