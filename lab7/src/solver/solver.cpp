#include <cfloat>

#include "../helpers.hpp"
#include "solver.hpp"

void Solver::solve()
{
    // const int block_i = grid.block_i(rank),
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

void Solver::boundary_layer_exchange()
{
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