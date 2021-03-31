#include <algorithm>

#include "grid.hpp"
#include "helpers.hpp"

size_t BlockGrid::cell_absolute_id(dim3<int> p) const
{
    return cell_absolute_id(p.x, p.y, p.z);
}

size_t BlockGrid::cell_absolute_id(int i, int j, int k) const
{
    return static_cast<size_t>(0
        + (k + 1) * ((bsize.x + 2) * (bsize.y + 2))
        + (j + 1) * (bsize.x + 2)
        + (i + 1));
}

dim3<int> BlockGrid::cell_idx(int n) const
{
    return {
        (n % ((bsize.x + 2) * (bsize.y + 2))) % (bsize.x + 2) - 1,
        (n % ((bsize.x + 2) * (bsize.y + 2))) / (bsize.x + 2) - 1,
        (n / ((bsize.x + 2) * (bsize.y + 2))) - 1
    };
}

size_t BlockGrid::cells_per_block() const
{
    return size_t((bsize.x + 2) * (bsize.y + 2) * (bsize.z + 2));
}

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

std::istream& operator>>(std::istream& in, Grid& grid)
{
    in >> grid.n_blocks >> grid.bsize;
    return in;
}

std::ostream& operator<<(std::ostream& out, const Grid& data)
{
    out
        << "rank: " << data.process_rank
        << " n_processes: " << data.n_processes
        << std::endl
        << data.n_blocks.print("n_blocks")
        << std::endl
        << data.bsize.print("block_size");
    return out;
}

int Grid::max_size() const { return std::max(bsize.x, std::max(bsize.y, bsize.z)); }

int Grid::block_absolute_id(dim3<int> p) const { return block_absolute_id(p.x, p.y, p.z); }
int Grid::block_absolute_id(int i, int j, int k) const
{
    return k * (n_blocks.x * n_blocks.y) + j * n_blocks.x + i;
}

dim3<int> Grid::block_idx() const
{
    return {
        (process_rank % (n_blocks.x * n_blocks.y)) % n_blocks.x,
        (process_rank % (n_blocks.x * n_blocks.y)) / n_blocks.x,
        (process_rank / (n_blocks.x * n_blocks.y))
    };
}

dim3<double> Grid::height(const dim3<double>& l_size) const
{
    dim3<double> res;

    res.x = l_size.x / double(n_blocks.x * bsize.x);
    res.y = l_size.y / double(n_blocks.y * bsize.y);
    res.z = l_size.z / double(n_blocks.z * bsize.z);

    return res;
}
