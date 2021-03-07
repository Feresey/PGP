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

    int max_size() const;

    int block_index(int i, int j, int k) const;
    int block_i(int n) const;
    int block_j(int n) const;
    int block_k(int n) const;

    int cell_idx(int i, int j, int k) const;
    int cell_i(int n) const;
    int cell_j(int n) const;
    int cell_k(int n) const;

    int cells_per_block() const;
    dim3<double> height(const dim3<double>& l_size) const;
};

#endif
