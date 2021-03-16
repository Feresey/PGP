#ifndef GRID_HPP
#define GRID_HPP

#include "dim3/dim3.hpp"

#include "helpers.cuh"

struct BlockGrid {
    mydim3<int> bsize;

    __host__ __device__ size_t cell_absolute_id(int i, int j, int k) const;
    __host__ __device__ size_t cell_absolute_id(mydim3<int> p) const;
    mydim3<int> cell_idx(int n) const;
    size_t cells_per_block() const;
};

struct Grid : BlockGrid {
    const int process_rank;
    const int n_processes;

    mydim3<int> bsize;
    mydim3<int> n_blocks;

    Grid(int process_rank, int n_processes);
    void mpi_bcast();

    friend std::istream& operator>>(std::istream& in, Grid& data);
    friend std::ostream& operator<<(std::ostream& out, const Grid& data);

    int max_size() const;

    int block_absolute_id(int i, int j, int k) const;
    mydim3<int> block_idx() const;

    //@ ну да, это скорее к Problem относится@
    mydim3<double> height(const mydim3<double>& l_size) const;
};

#endif
