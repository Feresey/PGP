#include "exchange.hpp"
#include "helpers.hpp"

Exchange::Exchange(const Grid& grid, const Task& task, GPU_pool& pool)
    : grid(grid)
    , task(task)
    , pool(pool)
{
    //@ да, так будет оверхед по памяти, но не нужно мучиться с границами.@
    const int max_dim = grid.max_size();

    this->send_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim + 2));
    this->receive_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim + 2));
}

void Exchange::exchange2D(
    const int block_idx, const int n_blocks, const int cell_size,
    const int a_size, const int b_size,
    const double lower_init, const double upper_init,
    const side_tag lower, const side_tag upper,
    std::function<size_t(int my, int a, int b)> get_cell_idx,
    std::function<int(int)> get_block_idx)
{
    const int count = a_size * b_size;

    MPI_Request requests[2];

    for (int each = 0; each <= 1; ++each) {
        const int copy_cell = (each == 0) ? 0 : (cell_size - 1);
        const double init_val = (each == 0) ? lower_init : upper_init;

        //@ Является ли текущий блок граничным@
        const bool is_boundary = block_idx == ((each == 0) ? 0 : (n_blocks - 1));

        const side_tag tag1 = (each == 0) ? lower : upper;
        const side_tag tag2 = (each == 0) ? upper : lower;
        if (!is_boundary) {
            const int exchange_process_rank = get_block_idx(block_idx + ((each == 0) ? -1 : 1));

            pool.load_gpu_border(tag1);

            //@ отсылка и прием нижнего граничного условия@
            for (int a = 0; a < a_size; ++a) {
                for (int b = 0; b < b_size; ++b) {
                    send_buffer[size_t(a * b_size + b)] = pool.data[get_cell_idx(copy_cell, a, b)];
                }
            }

            // CSC(MPI_Sendrecv(
            //     send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag1,
            //     receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag2,
            //     MPI_COMM_WORLD, MPI_STATUS_IGNORE));

            MPI_ERR(MPI_Isend(send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag1, MPI_COMM_WORLD, &requests[0]));
            MPI_ERR(MPI_Irecv(receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag2, MPI_COMM_WORLD, &requests[1]));

            MPI_ERR(MPI_Waitall(2, requests, MPI_STATUSES_IGNORE));
        }

        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                size_t idx = a * b_size + b;
                // debug("write cell idx: %ld", get_cell_idx(bound_cell, a, b));
                pool.data[idx] = (is_boundary) ? init_val : receive_buffer[idx];
                // std::cerr << pool.data[idx] << " ";
            }
            // std::cerr << std::endl;
        }
        // std::cerr << std::endl;
        // debug("mpi data end");

        pool.store_gpu_border(tag1);
    }
}

void Exchange::boundary_layer_exchange()
{
    const mydim3<int> block_idx = grid.block_idx();

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
    if (block_idx == 0) {
        pool.load_gpu_data();
        for (int i = 0; i < grid.bsize.x; ++i) {
            receive_buffer[size_t(i)] = pool.data[grid.cell_absolute_id(i, j, k)];
        }
    } else {
        int tag = k * grid.bsize.z + j;
        MPI_ERR(MPI_Recv(receive_buffer.data(), grid.bsize.x, MPI_DOUBLE, block_idx, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
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
                        if (bi != 0) {
                            out << " ";
                        }
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
    pool.load_gpu_data();
    for (int i = 0; i < grid.bsize.z; ++i) {
        for (int j = 0; j < grid.bsize.y; ++j) {
            for (int k = 0; k < grid.bsize.x; ++k) {
                send_buffer[size_t(k)] = pool.data[grid.cell_absolute_id(i, j, k)];
            }
            int tag = i * grid.bsize.z + j;
            MPI_Send(send_buffer.data(), grid.bsize.x, MPI_DOUBLE, ROOT_RANK, tag, MPI_COMM_WORLD);
        }
    }
}