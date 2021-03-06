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

std::ostream& operator<<(std::ostream& out, const Solver& solver)
{
    out << solver.grid
        << std::endl
        << solver.task
        << std::endl;
    return out;
}

void Solver::solve(Device& pool, const std::string& output)
{
    Exchange exchange(grid, task, pool);

    MPI_Barrier(MPI_COMM_WORLD);
    double error = 100.0;

    while (error > task.eps) {
        // debug("do iteration");
        exchange.boundary_layer_exchange();
        MPI_Barrier(MPI_COMM_WORLD);

        // debug("before calc");
        // pool.show(std::cerr);
        double local_error = pool.calc();
        // debug("after calc, error = %e", local_error);
        // pool.show(std::cerr);
        error = this->calc_error(local_error);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //*
    exchange.write_result(output);
    /*/
    if (grid.process_rank == ROOT_RANK) {
        std::fstream out(output, out.trunc | out.out);
        exchange.write_result(out);
    } else {
        exchange.send_result();
    }
    //*/
}

double Solver::calc_error(double local_error) const
{
    double all_error = 100.0;
    MPI_ERR(MPI_Allreduce(&local_error, &all_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    return all_error;
}