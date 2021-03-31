#include <cmath>
#include <sstream>

#include "exchange.hpp"
#include "helpers.hpp"
#include "sides.hpp"

Exchange::Exchange(const Grid& grid, const Task& task, Problem& problem)
    : grid(grid)
    , task(task)
    , problem(problem)
{
    //@ да, так будет оверхед по памяти, но не нужно мучиться с границами.@
    const int max_dim = grid.max_size();

    this->send_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim));
    this->receive_buffer = std::vector<double>(static_cast<size_t>(max_dim * max_dim));
}

void Exchange::exchange2D(
    const int block_idx, const int n_blocks, const int cell_size,
    const int a_size, const int b_size,
    const double lower_init, const double upper_init,
    const int recvtag_lower, const int recvtag_upper,
    std::function<size_t(int my, int a, int b)> get_cell_idx,
    std::function<int(int)> get_block_idx)
{
    const int count = a_size * b_size;

    MPI_Request req1, req2;

    for (int each = 0; each <= 1; ++each) {
        const int copy_cell = (each == 0) ? 0 : (cell_size - 1);
        const int bound_cell = (each == 0) ? -1 : cell_size;
        const double init_val = (each == 0) ? lower_init : upper_init;

        //@ Является ли текущий блок граничным@
        const bool is_boundary = block_idx == ((each == 0) ? 0 : (n_blocks - 1));

        if (!is_boundary) {
            const int tag1 = (each == 0) ? recvtag_lower : recvtag_upper;
            const int tag2 = (each == 0) ? recvtag_upper : recvtag_lower;
            const int exchange_process_rank = get_block_idx(block_idx + ((each == 0) ? -1 : 1));

            //@ отсылка и прием нижнего граничного условия@
            for (int a = 0; a < a_size; ++a) {
                for (int b = 0; b < b_size; ++b) {
                    send_buffer[size_t(a * b_size + b)] = problem.data[get_cell_idx(copy_cell, a, b)];
                }
            }

            // CSC(MPI_Sendrecv(
            //     send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag1,
            //     receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag2,
            //     MPI_COMM_WORLD, MPI_STATUS_IGNORE));

            // std::cerr << "I am " << grid.process_rank << ". send " << tag1 << std::endl;
            MPI_ERR(MPI_Isend(send_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag1, MPI_COMM_WORLD, &req1));
            // std::cerr << "I am " << grid.process_rank << ". receive " << tag2 << std::endl;
            MPI_ERR(MPI_Irecv(receive_buffer.data(), count, MPI_DOUBLE, exchange_process_rank, tag2, MPI_COMM_WORLD, &req2));

            MPI_ERR(MPI_Wait(&req1, MPI_STATUS_IGNORE));
            MPI_ERR(MPI_Wait(&req2, MPI_STATUS_IGNORE));
        }

        for (int a = 0; a < a_size; ++a) {
            for (int b = 0; b < b_size; ++b) {
                problem.data[get_cell_idx(bound_cell, a, b)] = (is_boundary) ? init_val : receive_buffer[size_t(a * b_size + b)];
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

void Exchange::write_result(const std::string& output)
{
    int n_outputs_per_block = grid.bsize.y * grid.bsize.z;
    dim3<int> block_idx = grid.block_idx();

    // одна строчка по x координате
    MPI_Datatype string_type;
    int len = snprintf(NULL, 0, "% e ", double(0.0));
    int str_size = len * grid.bsize.x;
    MPI_ERR(MPI_Type_contiguous(str_size, MPI_CHAR, &string_type));
    MPI_ERR(MPI_Type_commit(&string_type));

    MPI_Datatype pattern_type;
    // пусть строчки всегда будут по одной
    const std::vector<int> pattern_lens(size_t(n_outputs_per_block), 1);
    std::vector<MPI_Aint> pattern_disps(static_cast<size_t>(n_outputs_per_block));

    pattern_disps[0] = 0;
    for (int i = 1; i < n_outputs_per_block; ++i) {
        int increase = ((i % grid.bsize.y != 0) ? grid.n_blocks.x : (grid.n_blocks.x + (grid.n_blocks.y - 1) * grid.n_blocks.x * grid.bsize.y));
        pattern_disps[size_t(i)] = pattern_disps[size_t(i - 1)] + increase * str_size;
    }

    // ну оффсеты всегда кратны размеру string_type, так что хватило бы и MPI_Type_create_indexed.
    MPI_ERR(MPI_Type_create_hindexed(n_outputs_per_block, pattern_lens.data(), pattern_disps.data(), string_type, &pattern_type));
    MPI_ERR(MPI_Type_commit(&pattern_type));

    std::vector<char> res(static_cast<size_t>(str_size * grid.bsize.y * grid.bsize.z + 1), ' ');

    for (int k = 0; k < grid.bsize.z; ++k) {
        for (int j = 0; j < grid.bsize.y; ++j) {
            for (int i = 0; i < grid.bsize.x - 1; ++i) {
                int offset = len * (i + grid.bsize.x * j + (grid.bsize.x * grid.bsize.y) * k);
                sprintf(res.data() + offset, "% e ", problem.data[grid.cell_absolute_id(i, j, k)]);
            }
            int offset = len * ((grid.bsize.x - 1) + grid.bsize.x * j + (grid.bsize.x * grid.bsize.y) * k);
            if (block_idx.x == (grid.n_blocks.x - 1)) {
                sprintf(res.data() + offset, "% e\n", problem.data[grid.cell_absolute_id(grid.bsize.x - 1, j, k)]);
            } else {
                sprintf(res.data() + offset, "% e ", problem.data[grid.cell_absolute_id(grid.bsize.x - 1, j, k)]);
            }
        }
    }

    MPI_ERR(MPI_Barrier(MPI_COMM_WORLD));

    MPI_File fd;
    MPI_ERR(MPI_File_open(MPI_COMM_WORLD, output.data(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fd));

    int y_stride = grid.bsize.x * grid.n_blocks.x;
    int z_stride = y_stride * grid.bsize.y * grid.n_blocks.y;

    int global_y = block_idx.y * grid.bsize.y;
    int global_z = block_idx.z * grid.bsize.z;

    int disp = (z_stride * global_z + y_stride * global_y + block_idx.x * grid.bsize.x) / grid.bsize.x * str_size;
    MPI_ERR(MPI_File_set_errhandler(fd, MPI_ERRORS_ARE_FATAL));
    MPI_ERR(MPI_File_set_view(fd, disp, string_type, pattern_type, "native", MPI_INFO_NULL));
    // MPI_ERR(MPI_File_set_size(fd, 0));
    MPI_ERR(MPI_File_write(fd, res.data(), n_outputs_per_block, string_type, MPI_STATUS_IGNORE));
    MPI_ERR(MPI_Barrier(MPI_COMM_WORLD));

    MPI_ERR(MPI_File_close(&fd));
    MPI_ERR(MPI_Type_free(&string_type));
    MPI_ERR(MPI_Type_free(&pattern_type));
}
