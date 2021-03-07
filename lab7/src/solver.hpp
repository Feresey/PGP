#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <mpi.h>

#include <fstream>
#include <string>

#include "dim3.hpp"
#include "grid.hpp"

class Solver {
    int rank;
    int n_processes;

    Grid grid;

    double eps;
    dim3<double> l_size;
    double u_down, u_up, u_left, u_right, u_front, u_back;
    double u_0;

    std::string output_file_path;

    void read_data(std::istream& in);
    void show_data(std::ostream& out);
    void mpi_bcast();

public:
    Solver(std::istream& in);
};

#endif