#include "dim3.hpp"

template <>
MPI_Datatype dim3<int>::mpi_type() { return MPI_INT; }
template <>
MPI_Datatype dim3<double>::mpi_type() { return MPI_DOUBLE; }
