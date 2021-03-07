#include <mpi.h>

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "solver/solver.hpp"
#include "helpers.hpp"

int main(int argc, char** argv)
{
    CSC(MPI_Init(&argc, &argv));
    Solver s(std::cin);
    CSC(MPI_Finalize());
}