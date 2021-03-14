#ifndef DIM3_HPP
#define DIM3_HPP

#include <mpi.h>

#include <fstream>
#include <string>

#include "helpers.hpp"

template <class T>
class mydim3;

template <class T>
class dim3_print;

template <class T>
class mydim3 {
    MPI_Datatype mpi_type();

public:
    T x, y, z;

    mydim3(T x = 0, T y = 0, T z = 0)
        : x(x)
        , y(y)
        , z(z)
    {
    }
    mydim3(std::istream& in) { in >> x >> y >> z; }
    friend std::istream& operator>>(std::istream& in, mydim3<T>& val)
    {
        val = mydim3<T>(in);
        return in;
    }
    dim3_print<T> print(const std::string& prefix) const { return dim3_print<T>(*this, prefix); }
    void mpi_bcast()
    {
        bcast(&x, this->mpi_type());
        bcast(&y, this->mpi_type());
        bcast(&z, this->mpi_type());
    }
};

template <class T>
class dim3_print {
    const mydim3<T>& data;
    const std::string& prefix;

public:
    dim3_print(const mydim3<T>& data, const std::string& prefix)
        : data(data)
        , prefix(prefix)
    {
    }
    friend std::ostream& operator<<(std::ostream& out, const dim3_print<T>& data)
    {
        out
            << data.prefix
            << ": ("
            << data.data.x << ","
            << data.data.y << ","
            << data.data.z
            << ")";
        return out;
    }
};

#endif