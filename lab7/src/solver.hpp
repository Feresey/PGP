#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "grid/grid.hpp"

#include "problem.hpp"

class Solver {
    const Grid& grid;
    const Task& task;

    // friend std::istream& operator>>(std::istream& in, Solver& data);
    friend std::ostream& operator<<(std::ostream& out, const Solver& data);

    void boundary_layer_exchange();
    double calc_error(double local_err) const;

public:
    Solver(const Grid& grid, const Task& task);
    void solve(Problem& problem, const std::string& output);
};

#endif