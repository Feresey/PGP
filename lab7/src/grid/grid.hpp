#ifndef GRID_HPP
#define GRID_HPP

#include "dim3/dim3.hpp"

struct Grid {
    const int process_rank;
    const int n_processes;

    dim3<int> bsize;
    dim3<int> n_blocks;

    Grid(int process_rank, int n_processes);
    void mpi_bcast();

    friend std::istream& operator>>(std::istream& in, Grid& data);
    friend std::ostream& operator<<(std::ostream& out, const Grid& data);

    int max_size() const;

    int block_absolute_id(int i, int j, int k) const;
    dim3<int> block_idx() const;

    size_t cell_absolute_id(int i, int j, int k) const;

    dim3<int> cell_idx(int n) const;

    size_t cells_per_block() const;

    // ну да, это скорее к Problem относится
    dim3<double> height(const dim3<double>& l_size) const;
};

#endif
