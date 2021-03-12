#include <cfloat>

#include "helpers.hpp"
#include "solver.hpp"

std::istream& operator>>(std::istream& in, Task& task)
{
    in
        >> task.eps
        >> task.l_size
        >> task.u_bottom >> task.u_top
        >> task.u_left >> task.u_right
        >> task.u_front >> task.u_back
        >> task.u_0;
    return in;
}

std::ostream& operator<<(std::ostream& out, const Task& task)
{
    out << "eps:\t" << task.eps
        << std::endl
        << task.l_size.print("l_size") << std::endl
        << "u_bottom: " << task.u_bottom
        << " u_top: " << task.u_top
        << " u_left: " << task.u_left
        << " u_right: " << task.u_right
        << " u_front: " << task.u_front
        << " u_back: " << task.u_back
        << std::endl
        << "u_0: " << task.u_0;
    return out;
}

std::ostream& operator<<(std::ostream& out, const Solver& solver)
{
    out << solver.grid
        << std::endl
        << solver.task
        << std::endl;
    return out;
}

void Task::mpi_bcast()
{
    l_size.mpi_bcast();
    bcast_double(&eps);
    bcast_double(&u_top);
    bcast_double(&u_bottom);
    bcast_double(&u_left);
    bcast_double(&u_right);
    bcast_double(&u_front);
    bcast_double(&u_back);
    bcast_double(&u_0);
}
