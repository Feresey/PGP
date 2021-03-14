#include "dim3.hpp"

template <>
MPI_Datatype mydim3<int>::mpi_type() { return MPI_INT; }
template <>
MPI_Datatype mydim3<double>::mpi_type() { return MPI_DOUBLE; }
