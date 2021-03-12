#ifndef DIM3_HPP
#define DIM3_HPP

#include <mpi.h>

#include <fstream>
#include <string>

#include "helpers.hpp"

template <class T>
class dim3;

template <class T>
class dim3_print;

template <class T>
class dim3 {
    MPI_Datatype mpi_type();
public:
    T x, y, z;

    dim3(T x = 0, T y = 0, T z = 0)
        : x(x)
        , y(y)
        , z(z)
    {
    }
    dim3(std::istream& in) { in >> x >> y >> z; }
    friend std::istream& operator>>(std::istream& in, dim3<T>& val)
    {
        val = dim3<T>(in);
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
    const dim3<T>& data;
    const std::string& prefix;

public:
    dim3_print(const dim3<T>& data, const std::string& prefix)
        : data(data)
        , prefix(prefix)
    {
    }
    friend std::ostream& operator<<(std::ostream& out, const dim3_print<T>& data)
    {
        out
            << data.prefix + "_x:\t" << data.data.x << "\t"
            << data.prefix + "_y:\t" << data.data.y << "\t"
            << data.prefix + "_z:\t" << data.data.z;
        return out;
    }
};

#endif