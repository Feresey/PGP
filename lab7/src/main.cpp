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

    Grid grid(rank, n_processes);
    Task task;
    std::string output_file_path;
    std::cin
        >> grid
        >> output_file_path
        >> task;


    grid.mpi_bcast();
    task.mpi_bcast();

    Problem problem(task, grid);
    Solver solver(grid, task);

    solver.solve(problem);

    // Solver s(rank, n_processes, std::cin);
    CSC(MPI_Finalize());
}