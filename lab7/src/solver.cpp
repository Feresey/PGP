#include "solver.hpp"

#define ROOT_RANK 0

Solver::Solver(std::istream& in)
{
    CSC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CSC(MPI_Comm_size(MPI_COMM_WORLD, &n_processes));

    if (rank == ROOT_RANK) {
        this->read_data(in);
    }

    this->mpi_bcast();
}

void Solver::read_data(std::istream& in)
{
    in >> n_blocks;
    in >> block_size;
    in >> output_file_path;
    in >> eps;
    in >> l_size;
    in >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back;
    in >> u_0;
}

void Solver::show_data(std::ostream& out)
{
    out << n_blocks.print("n_blocks")
        << std::endl
        << block_size.print("block_size")
        << std::endl
        << "output_file_path: " << output_file_path
        << std::endl
        << "eps: " << eps
        << std::endl
        << l_size.print("l_size") << std::endl
        << "u_down: " << u_down
        << "u_up: " << u_up
        << "u_left: " << u_left
        << "u_right: " << u_right
        << "u_front: " << u_front
        << "u_back: " << u_back
        << std::endl
        << "u_0: " << u_0
        << std::endl;
}

inline void bcast_int(int* val)
{
    CSC(MPI_Bcast(val, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
}

inline void bcast_double(double* val)
{
    CSC(MPI_Bcast(val, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
}

void Solver::mpi_bcast()
{
    bcast_int(&n_blocks.x);
    bcast_int(&n_blocks.y);
    bcast_int(&n_blocks.z);
    bcast_int(&block_size.x);
    bcast_int(&block_size.y);
    bcast_int(&block_size.z);
    bcast_double(&eps);
    bcast_double(&l_size.x);
    bcast_double(&l_size.y);
    bcast_double(&l_size.z);
    bcast_double(&u_down);
    bcast_double(&u_up);
    bcast_double(&u_left);
    bcast_double(&u_right);
    bcast_double(&u_front);
    bcast_double(&u_back);
    bcast_double(&u_0);
}