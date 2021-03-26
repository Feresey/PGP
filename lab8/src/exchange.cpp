#include "exchange.hpp"
#include "helpers.hpp"

Exchange::Exchange(const Grid& grid, const Task& task, Device& pool)
    : grid(grid)
    , task(task)
    , pool(pool)
{
    //@ да, так будет оверхед по памяти, но не нужно мучиться с границами.@
    const int max_dim = grid.max_size();

    this->send_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim + 2));
    this->receive_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim + 2));
}

void Exchange::exchange2D(dim3_type block_coord)
{
    double lower_init, upper_init;
    side_tag lower_tag, upper_tag;

    switch (dim3_type_to_layer_tag(block_coord)) {
    default:
    case LEFT_RIGHT:
        lower_tag = LEFT;
        lower_init = task.u_left;
        upper_tag = RIGHT;
        upper_init = task.u_right;
        break;
    case FRONT_BACK:
        lower_tag = FRONT;
        lower_init = task.u_front;
        upper_tag = BACK;
        upper_init = task.u_back;
        break;
    case VERTICAL:
        lower_tag = BOTTOM;
        lower_init = task.u_bottom;
        upper_tag = TOP;
        upper_init = task.u_top;
        break;
    }

    std::pair<int,int> sizes = other_sizes(grid, dim3_type_to_layer_tag(block_coord));
    int a_size = sizes.first, b_size = sizes.second;
    // for (auto elem = grid.bsize.begin(); elem != grid.bsize.end(); ++elem) {
    //     if (elem.get_type() == block_coord) {
    //         continue;
    //     }
    //     if (a_size == -1) {
    //         a_size = *elem;
    //     } else {
    //         b_size = *elem;
    //     }
    // }
    // debug("sizes (%d,%d)", a_size, b_size);

    const int count = a_size * b_size;
    const mydim3<int> block_absolute_idx = grid.block_idx();
    const int block_idx = block_absolute_idx[block_coord];

    MPI_Request requests[2];

    for (int each = 0; each <= 1; ++each) {
        const double init_val = (each == 0) ? lower_init : upper_init;

        //@ Является ли текущий блок граничным@
        const bool is_boundary = block_idx == ((each == 0) ? 0 : (grid.n_blocks[block_coord] - 1));

        const side_tag tag1 = (each == 0) ? lower_tag : upper_tag;
        const side_tag tag2 = (each == 0) ? upper_tag : lower_tag;
        if (!is_boundary) {
            mydim3<int> exchange_block = block_absolute_idx;
            exchange_block[block_coord] = block_idx + ((each == 0) ? -1 : 1);
            const int exchange_process_rank = grid.block_absolute_id(exchange_block);
            // debug("load border %d", tag1);
            pool.load_gpu_border(tag1);

            //@ отсылка и прием нижнего граничного условия@
            for (int a = 0; a < a_size; ++a) {
                for (int b = 0; b < b_size; ++b) {
                    size_t idx = size_t(a * b_size + b);
                    send_buffer[idx] = pool.data[idx];
                    // std::cerr << pool.data[idx] << " ";
                }
                // std::cerr << std::endl;
            }
            // std::cerr << std::endl;

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
                size_t idx = size_t(a * b_size + b);
                pool.data[idx] = (is_boundary) ? init_val : receive_buffer[idx];
            }
        }

        pool.store_gpu_border(tag1);
        // debug("show all data");
        // pool.show(std::cerr);
    }
}

void Exchange::boundary_layer_exchange()
{
    exchange2D(DIM3_TYPE_Z);
    // MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ASSERT);
    exchange2D(DIM3_TYPE_Y);
    exchange2D(DIM3_TYPE_X);
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
    if (grid.process_rank != ROOT_RANK) {
        send_result();
        return;
    }
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