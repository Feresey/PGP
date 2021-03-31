#include <cfloat>
#include <functional>

#include "exchange.hpp"
#include "helpers.hpp"
#include "solver.hpp"

Solver::Solver(const Grid& grid, const Task& task)
    : grid(grid)
    , task(task)
{
}

void Solver::solve(Problem& problem, const std::string& output)
{
    Exchange exchange(grid, task, problem);

    MPI_Barrier(MPI_COMM_WORLD);
    double error = DBL_MAX;
    while (error > task.eps) {
        exchange.boundary_layer_exchange();
        MPI_Barrier(MPI_COMM_WORLD);

        double local_error = problem.calc();
        // exchange.write_result(std::cerr);
        error = this->calc_error(local_error);
        // debug("after calc error: %e", local_error);
        // MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ASSERT);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    exchange.write_result(output);
}

double Solver::calc_error(double local_error) const
{
    double all_error = DBL_MAX;
    MPI_ERR(MPI_Allreduce(&local_error, &all_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    return all_error;
}