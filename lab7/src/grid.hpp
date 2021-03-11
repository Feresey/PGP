#ifndef GRID_HPP
#define GRID_HPP

#include "dim3.hpp"

struct Grid {
    dim3<int> bsize;
    dim3<int> n_blocks;

    Grid();
    Grid(std::istream& in);
    void read_data(std::istream& in);
    void show_data(std::ostream& out) const;
    void mpi_bcast();

    friend std::ostream& operator<<(std::ostream& out, const Grid& data);
    friend std::istream& operator>>(std::istream& in, Grid& data);

    uint max_size() const;

    uint block_idx(int i, int j, int k) const;
    uint block_i(int n) const;
    uint block_j(int n) const;
    uint block_k(int n) const;

    uint cell_idx(int i, int j, int k) const;
    uint cell_i(int n) const;
    uint cell_j(int n) const;
    uint cell_k(int n) const;

    uint cells_per_block() const;

    // ну да, это скорее к Problem относится
    dim3<double> height(const dim3<double>& l_size) const;
};

#endif
