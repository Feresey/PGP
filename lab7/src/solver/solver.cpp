#include <cfloat>

#include "../helpers.hpp"
#include "solver.hpp"

void Solver::solve()
{
    // const int block_i = grid.block_i(rank),
    //           block_j = grid.block_j(rank),
    //           block_k = grid.block_k(rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double max_error = DBL_MAX;
    while (max_error > eps) {
        this->boundary_layer_exchange();
        MPI_Barrier(MPI_COMM_WORLD);

        double local_error = this->problem.calc();
        max_error = this->calc_error(local_error);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    this->write_result();
}

void Solver::boundary_layer_exchange() {

}

void Solver::write_result() {

}

double Solver::calc_error(double local_error) const{
    return local_error;
}