#ifndef GRID_HPP
#define GRID_HPP

#include "dim3.hpp"

class Grid {
    dim3<int> block_size;
    dim3<int> n_blocks;

public:
    Grid();
    Grid(std::istream& in);
    void read_data(std::istream& in);
    void show_data(std::ostream& out) const;
    void mpi_bcast();

    friend std::ostream& operator<<(std::ostream& out, const Grid& data);
    friend std::istream& operator>>(std::istream& in, Grid& data);
};

#endif
