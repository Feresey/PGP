#include <mpi.h>

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "grid/grid.hpp"
#include "helpers.hpp"
#include "solver.hpp"

int main(int argc, char** argv)
{
    CSC(MPI_Init(&argc, &argv));
    int rank, n_processes;
    CSC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CSC(MPI_Comm_size(MPI_COMM_WORLD, &n_processes));
    CSC(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    Grid grid(rank, n_processes);
    Task task;
    std::string output;
    std::cin
        >> grid
        >> output
        >> task;

    if (rank == ROOT_RANK) {
        if (grid.n_blocks.x * grid.n_blocks.y * grid.n_blocks.z != n_processes) {
            std::cerr
                << "incorrect block dimensions. actual " << grid.n_blocks.print("dim")
                << ", but got n_processes: " << n_processes
                << std::endl;
            CSC(MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    grid.mpi_bcast();
    task.mpi_bcast();

    Problem problem(task, grid);
    Solver solver(grid, task);

    std::cout << solver << std::endl;

    std::cout.flush();

    solver.solve(problem, output);

    CSC(MPI_Finalize());
}