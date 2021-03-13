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
        error = this->calc_error(local_error);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::fstream out(output, out.trunc | out.out);
    exchange.write_result(out);
}

double Solver::calc_error(double local_error) const
{
    double all_error = DBL_MAX;
    CSC(MPI_Allreduce(&local_error, &all_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    return all_error;
}