#include "grid.hpp"
#include "helpers.hpp"

Grid::Grid()
    : bsize()
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
    in >> bsize >> n_blocks;
}

void Grid::show_data(std::ostream& out) const
{
    out
        << bsize.print("block_size")
        << std::endl
        << n_blocks.print("n_blocks");
}

void Grid::mpi_bcast()
{
    bsize.mpi_bcast();
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

int Grid::max_size() const { return std::max(bsize.x, std::max(bsize.y, bsize.z)); }

// боже, какая жесть

int Grid::block_index(int i, int j, int k) const { return k * (n_blocks.x * n_blocks.y) + j * n_blocks.x + i; }
int Grid::block_i(int n) const { return (n % (n_blocks.x * n_blocks.y)) % n_blocks.x; }
int Grid::block_j(int n) const { return (n % (n_blocks.x * n_blocks.y)) / n_blocks.x; }
int Grid::block_k(int n) const { return n / (n_blocks.x * n_blocks.y); }

int Grid::cell_idx(int i, int j, int k) const
{
    return (k + 1) * ((bsize.x + 2) * (bsize.y + 2))
        + (j + 1) * (bsize.x + 2)
        + (i + 1);
}
int Grid::cell_i(int n) const { return (n % ((bsize.x + 2) * (bsize.y + 2))) % (bsize.x + 2) - 1; }
int Grid::cell_j(int n) const { return (n % ((bsize.x + 2) * (bsize.y + 2))) / (bsize.x + 2) - 1; }
int Grid::cell_k(int n) const { return (n / ((bsize.x + 2) * (bsize.y + 2))) - 1; }

// TODO
int Grid::cells_per_block() const { return -1; }
dim3<double> Grid::height(const dim3<double>& l_size) const { return {}; }
