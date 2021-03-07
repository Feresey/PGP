#include "solver.hpp"
#include "helpers.hpp"

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
    in
        >> grid
        >> output_file_path
        >> eps
        >> l_size
        >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back
        >> u_0;
}

void Solver::show_data(std::ostream& out)
{
    out << grid
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

void Solver::mpi_bcast()
{
    grid.mpi_bcast();
    l_size.mpi_bcast();
    bcast_double(&eps);
    bcast_double(&u_down);
    bcast_double(&u_up);
    bcast_double(&u_left);
    bcast_double(&u_right);
    bcast_double(&u_front);
    bcast_double(&u_back);
    bcast_double(&u_0);
}