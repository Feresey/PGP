#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <fstream>
#include <string>
#include <vector>

#include "../grid.hpp"

#include "problem.hpp"

class Solver {
    int rank;
    int n_processes;

    Grid grid;

    dim3<double> l_size;
    double eps;
    double u_down, u_up, u_left, u_right, u_front, u_back;
    double u_0;

    std::string output_file_path;

    std::vector<double> send_buffer;

    Problem problem;

    void read_data(std::istream& in);
    void show_data(std::ostream& out) const;
    void mpi_bcast();

    void boundary_layer_exchange();
    double calc_error(double local_err) const;

    void write_result();

    void exchange2D(
        const uint block_idx, const uint block_size,
        const uint a_size, const uint b_size,
        const double lower_init, const double upper_init,
        const int recvtag_lower, const int recvtag_upper);

public:
    Solver(std::istream& in);
    void solve();
};

#endif