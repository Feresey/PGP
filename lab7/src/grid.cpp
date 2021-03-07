#include "grid.hpp"
#include "helpers.hpp"

Grid::Grid()
    : block_size()
    , n_blocks()
{
}

Grid::Grid(std::istream& in)
    : Grid()
{
    this->read_data(in);
}

void Grid::read_data(std::istream& in)
{
    in >> block_size >> n_blocks;
}

void Grid::show_data(std::ostream& out) const
{
    out
        << block_size.print("block_size")
        << std::endl
        << n_blocks.print("n_blocks");
}

void Grid::mpi_bcast()
{
    block_size.mpi_bcast();
    n_blocks.mpi_bcast();
}

std::ostream& operator<<(std::ostream& out, const Grid& data)
{
    data.show_data(out);
    return out;
}

std::istream& operator>>(std::istream& in, Grid& data)
{
    data = Grid(in);
    return in;
}