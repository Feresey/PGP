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
    int fuck = 2;
    while (error > task.eps) {
        exchange.boundary_layer_exchange();
        MPI_Barrier(MPI_COMM_WORLD);

        std::cerr << "before calc" << std::endl;
        problem.show(std::cerr);
        exchange.write_result(std::cerr);
        double local_error = problem.calc();
        error = this->calc_error(local_error);
        std::cerr << "after calc, error " << local_error << std::endl;
        // problem.show(std::cerr);
        // exchange.write_result(std::cerr);
        // if (--fuck == 0) {
        //     MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ASSERT);
        // }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (grid.process_rank == ROOT_RANK) {
        std::fstream out(output, out.trunc | out.out);
        exchange.write_result(out);
    } else {
        exchange.send_result();
    }
}

double Solver::calc_error(double local_error) const
{
    double all_error = DBL_MAX;
    MPI_ERR(MPI_Allreduce(&local_error, &all_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    return all_error;
}