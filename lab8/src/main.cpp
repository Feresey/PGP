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
#include "pool/pool.hpp"
#include "pool/task.hpp"
#include "solver.hpp"

#include <chrono>

int main(int argc, char** argv)
{
    MPI_ERR(MPI_Init(&argc, &argv));
    int rank, n_processes;
    MPI_ERR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_ERR(MPI_Comm_size(MPI_COMM_WORLD, &n_processes));
    MPI_ERR(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    Grid grid(rank, n_processes);
    Task task;
    std::string output(PATH_MAX, '\0');
    if (rank == ROOT_RANK) {
        std::cin >> grid;
        output.resize(0);
        std::cin >> output;
        std::cin >> task;
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
    MPI_ERR(MPI_Bcast((void*)output.data(), PATH_MAX, MPI_CHAR, ROOT_RANK, MPI_COMM_WORLD));
    task.mpi_bcast();

    MPI_Comm local_comm;
    int local_rank;
    MPI_ERR(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    MPI_ERR(MPI_Comm_rank(local_comm, &local_rank));
    MPI_ERR(MPI_Comm_free(&local_comm));

    int n_devices;
    getDeviceCount(&n_devices);
    setDevice(local_rank % n_devices);

    Device device = Device(grid, task);
    Solver solver(grid, task);

    if (rank == ROOT_RANK) {
        std::cerr << solver << std::endl;
    }

    using scl = std::chrono::system_clock;
    auto start = scl::now();
    solver.solve(device, output);
    if (rank == ROOT_RANK) {
        auto end = scl::now();
        std::chrono::duration<double> secs = end - start;
        std::cerr << secs.count() * 1000.0 << std::endl;
    }

    MPI_ERR(MPI_Finalize());
}