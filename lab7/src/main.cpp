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

#include <time.h>

// call this function to start a nanosecond-resolution timer
struct timespec timer_start()
{
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

typedef unsigned long long ull;

// call this function to end a timer, returning nanoseconds elapsed as a long
ull timer_end(struct timespec start_time)
{
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    ull diffInNanos = (ull)(end_time.tv_sec - start_time.tv_sec) * (ull)1e9 + (ull)(end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}

int main(int argc, char** argv)
{
    MPI_ERR(MPI_Init(&argc, &argv));
    int rank, n_processes;
    MPI_ERR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_ERR(MPI_Comm_size(MPI_COMM_WORLD, &n_processes));
    MPI_ERR(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

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
            MPI_ERR(MPI_Abort(MPI_COMM_WORLD, MPI_ERR_DIMS));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    grid.mpi_bcast();
    task.mpi_bcast();

    Problem problem(task, grid);
    Solver solver(grid, task);

    std::cout << solver << std::endl;

    std::cout.flush();

    struct timespec start_time = timer_start();
    solver.solve(problem, output);
    ull res = timer_end(start_time);
    if (rank == ROOT_RANK) {
        std::cerr << res << std::endl;
    }

    MPI_ERR(MPI_Finalize());
}