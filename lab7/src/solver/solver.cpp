#include <cfloat>
#include <functional>

#include "../helpers.hpp"
#include "solver.hpp"

void Solver::solve()
{
    // const int block_idx = grid.block_idx(rank),
    //           block_j = grid.block_j(rank),
    //           block_k = grid.block_k(rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double error = DBL_MAX;
    while (error > eps) {
        this->boundary_layer_exchange();
        MPI_Barrier(MPI_COMM_WORLD);

        double local_error = this->problem.calc();
        error = this->calc_error(local_error);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    this->write_result();
}

// TODO

void Solver::exchange2D(
    const uint block_idx, const uint block_size,
    const uint a_size, const uint b_size,
    const double lower_init, const double upper_init,
    const int recvtag_lower, const int recvtag_upper)
{
    if (block_idx > 0) {
        // отсылка и прием левого граничного условия
        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < b_size; ++b)
                send_buffer[a * b_size + b] = problem.get_result()[grid.cell_idx(0, a, b)];

        int count = a_size * b_size;
        int exchange_process_rank = grid.block_idx(block_idx - 1, block_j, block_k);

        // TODO
        MPI_Sendrecv(send_buffer, count, MPI_DOUBLE, exchange_process_rank, recvtag_lower,
            send_buffer, count, MPI_DOUBLE, exchange_process_rank, recvtag_upper,
            MPI_COMM_WORLD, &status);

        for (int a = 0; a < a_size; ++a)
            for (int b = 0; b < b_size; ++b)
                u[grid.cell_idx(-1, a, b)] = buffer1[a * block_size_z + b];
    } else {
        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(-1, j, k)] = u_left;
    }

    if (block_idx < block_size - 1) {
        // отсылка и прием правого граничного условия
        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                buffer[j * block_size_z + k] = u[grid.cell_idx(block_size_x - 1, j, k)];

        int count = block_size_y * block_size_z;
        int exchange_process_rank = block_index(block_idx + 1, block_j, block_k);

        MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, RIGHT,
            buffer1, count, MPI_DOUBLE, exchange_process_rank, LEFT,
            MPI_COMM_WORLD, &status);

        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(block_size_x, j, k)] = buffer1[j * block_size_z + k];
    } else {
        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(block_size_x, j, k)] = u_right;
    }
}

inline void fill_buffer(
    std::vector<double>& buffer,
    const std::vector<double>& src,
    const uint a_size, const uint b_size,
    std::function<int(int, int)> get_idx)
{
    for (int a = 0; a < a_size; ++a)
        for (int b = 0; b < b_size; ++b)
            buffer[a * b_size + b] = src[get_idx(a, b)];
}

void Solver::boundary_layer_exchange()
{
    if (block_idx > 0) {
        // отсылка и прием левого граничного условия
        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                buffer[j * block_size_z + k] = u[grid.cell_idx(0, j, k)];

        int count = block_size_y * block_size_z;
        int exchange_process_rank = block_index(block_idx - 1, block_j, block_k);

        MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, LEFT,
            buffer1, count, MPI_DOUBLE, exchange_process_rank, RIGHT,
            MPI_COMM_WORLD, &status);

        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(-1, j, k)] = buffer1[j * block_size_z + k];
    } else {
        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(-1, j, k)] = u_left;
    }

    if (block_idx < n_blocks_x - 1) {
        // отсылка и прием правого граничного условия
        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                buffer[j * block_size_z + k] = u[grid.cell_idx(block_size_x - 1, j, k)];

        int count = block_size_y * block_size_z;
        int exchange_process_rank = block_index(block_idx + 1, block_j, block_k);

        MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, RIGHT,
            buffer1, count, MPI_DOUBLE, exchange_process_rank, LEFT,
            MPI_COMM_WORLD, &status);

        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(block_size_x, j, k)] = buffer1[j * block_size_z + k];
    } else {
        for (int j = 0; j < block_size_y; ++j)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(block_size_x, j, k)] = u_right;
    }

    if (block_j > 0) {
        // отсылка и прием переднего граничного условия
        for (int i = 0; i < block_size_x; ++i)
            for (int k = 0; k < block_size_z; ++k)
                buffer[i * block_size_z + k] = u[grid.cell_idx(i, 0, k)];

        int count = block_size_x * block_size_z;
        int exchange_process_rank = block_index(block_idx, block_j - 1, block_k);

        MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, FRONT,
            buffer1, count, MPI_DOUBLE, exchange_process_rank, BACK,
            MPI_COMM_WORLD, &status);

        for (int i = 0; i < block_size_x; ++i)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(i, -1, k)] = buffer1[i * block_size_z + k];
    } else {
        for (int i = 0; i < block_size_x; ++i)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(i, -1, k)] = u_front;
    }

    if (block_j < n_blocks_y - 1) {
        // отсылка и прием заднего граничного условия
        for (int i = 0; i < block_size_x; ++i)
            for (int k = 0; k < block_size_z; ++k)
                buffer[i * block_size_z + k] = u[grid.cell_idx(i, block_size_y - 1, k)];

        int count = block_size_x * block_size_z;
        int exchange_process_rank = block_index(block_idx, block_j + 1, block_k);

        MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, BACK,
            buffer1, count, MPI_DOUBLE, exchange_process_rank, FRONT,
            MPI_COMM_WORLD, &status);

        for (int i = 0; i < block_size_x; ++i)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(i, block_size_y, k)] = buffer1[i * block_size_z + k];
    } else {
        for (int i = 0; i < block_size_x; ++i)
            for (int k = 0; k < block_size_z; ++k)
                u[grid.cell_idx(i, block_size_y, k)] = u_back;
    }

    if (block_k > 0) {
        // отсылка и прием нижнего граничного условия
        for (int i = 0; i < block_size_x; ++i)
            for (int j = 0; j < block_size_y; ++j)
                buffer[i * block_size_y + j] = u[grid.cell_idx(i, j, 0)];

        int count = block_size_x * block_size_y;
        int exchange_process_rank = block_index(block_idx, block_j, block_k - 1);

        MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, DOWN,
            buffer1, count, MPI_DOUBLE, exchange_process_rank, UP,
            MPI_COMM_WORLD, &status);

        for (int i = 0; i < block_size_x; ++i)
            for (int j = 0; j < block_size_y; ++j)
                u[grid.cell_idx(i, j, -1)] = buffer1[i * block_size_y + j];
    } else {
        for (int i = 0; i < block_size_x; ++i)
            for (int j = 0; j < block_size_y; ++j)
                u[grid.cell_idx(i, j, -1)] = u_down;
    }

    if (block_k < n_blocks_z - 1) {
        // отсылка и прием верхнего граничного условия
        for (int i = 0; i < block_size_x; ++i)
            for (int j = 0; j < block_size_y; ++j)
                buffer[i * block_size_y + j] = u[grid.cell_idx(i, j, block_size_z - 1)];

        int count = block_size_x * block_size_y;
        int exchange_process_rank = block_index(block_idx, block_j, block_k + 1);

        MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, UP,
            buffer1, count, MPI_DOUBLE, exchange_process_rank, DOWN,
            MPI_COMM_WORLD, &status);

        for (int i = 0; i < block_size_x; ++i)
            for (int j = 0; j < block_size_y; ++j)
                u[grid.cell_idx(i, j, block_size_z)] = buffer1[i * block_size_y + j];
    } else {
        for (int i = 0; i < block_size_x; ++i)
            for (int j = 0; j < block_size_y; ++j)
                u[grid.cell_idx(i, j, block_size_z)] = u_up;
    }
}

void Solver::write_result()
{
}

double Solver::calc_error(double local_error) const
{
    double all_error = DBL_MAX;
    CSC(MPI_Allreduce(&local_error, &all_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    return all_error;
}