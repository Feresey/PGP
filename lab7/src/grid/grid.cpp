#include "grid.hpp"
#include "helpers.hpp"

Grid::Grid(int process_rank, int n_processes)
    : process_rank(process_rank)
    , n_processes(n_processes)
{
}

void Grid::mpi_bcast()
{
    bsize.mpi_bcast();
    n_blocks.mpi_bcast();
}

std::ostream& operator<<(std::ostream& out, const Grid& data)
{
    out
        << data.bsize.print("block_size")
        << std::endl
        << data.n_blocks.print("n_blocks");
    return out;
}

std::istream& operator>>(std::istream& in, Grid& grid)
{
    in >> grid.bsize >> grid.n_blocks;
    return in;
}

int Grid::max_size() const { return std::max(bsize.x, std::max(bsize.y, bsize.z)); }

// боже, какая жесть

int Grid::block_idx(int i, int j, int k) const
{
    return k * (n_blocks.x * n_blocks.y) + j * n_blocks.x + i;
}
int Grid::block_i() const
{
    return (process_rank % (n_blocks.x * n_blocks.y)) % n_blocks.x;
}

int Grid::block_j() const
{
    return (process_rank % (n_blocks.x * n_blocks.y)) / n_blocks.x;
}
int Grid::block_k() const
{
    return process_rank / (n_blocks.x * n_blocks.y);
}

size_t Grid::cell_idx(int i, int j, int k) const
{
    return static_cast<size_t>((k + 1) * ((bsize.x + 2) * (bsize.y + 2))
        + (j + 1) * (bsize.x + 2)
        + (i + 1));
}
// int Grid::cell_i(int n) const { return (n % ((bsize.x + 2) * (bsize.y + 2))) % (bsize.x + 2) - 1; }
// int Grid::cell_j(int n) const { return (n % ((bsize.x + 2) * (bsize.y + 2))) / (bsize.x + 2) - 1; }
// int Grid::cell_k(int n) const { return (n / ((bsize.x + 2) * (bsize.y + 2))) - 1; }

size_t Grid::cells_per_block() const
{
    return static_cast<size_t>((bsize.x + 2) * (bsize.y + 2) * (bsize.z + 2));
}

dim3<double> Grid::height(const dim3<double>& l_size) const
{
    dim3<double> res;

    res.x = l_size.x / (n_blocks.x * bsize.x);
    res.y = l_size.y / (n_blocks.y * bsize.y);
    res.z = l_size.z / (n_blocks.z * bsize.z);

    return res;
}
