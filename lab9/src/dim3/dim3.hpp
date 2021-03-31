#ifndef DIM3_HPP
#define DIM3_HPP

#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include "helpers.hpp"

template <class T>
class dim3;

template <class T>
class dim3_print;

enum dim3_type {
    DIM3_TYPE_X,
    DIM3_TYPE_Y,
    DIM3_TYPE_Z
};

template <class T>
class dim3 {
    MPI_Datatype mpi_type();

public:
    template <class I, class Out = T>
    class iterator {
        I data;
        int index;

    public:
        // iterator traits
        using difference_type = dim3_type;
        using value_type = Out;
        using pointer = Out*;
        using reference = Out&;
        using iterator_category = std::random_access_iterator_tag;

        iterator(I data, bool end = false)
            : data(data)
            , index(DIM3_TYPE_X)
        {
            if (end) {
                index = 3;
            }
        }

        iterator& operator++()
        {
            ++index;
            return *this;
        }

        difference_type get_type() const { return difference_type(index); }

        reference operator*() { return data->operator[](difference_type(index)); }

        difference_type operator-(const iterator& rhs) { return difference_type(this->index - rhs.index); }
        bool operator==(const iterator& rhs) { return this->index == rhs.index; }
        bool operator!=(const iterator& rhs) { return this->index != rhs.index; }
    };

    T x, y, z;

    dim3(T x = {}, T y = {}, T z = {})
        : x(x)
        , y(y)
        , z(z)
    {
    }
    dim3(std::istream& in) { in >> x >> y >> z; }

    dim3(int my, dim3_type type, int a, int b)
    {
        int index = 0;
        const auto start = begin(), end = this->end();
        for (auto elem = begin(); elem != end; ++elem) {
            dim3_type elem_type = std::distance(start, elem);
            if (elem_type == type) {
                this[type] = my;
            } else {
                this[elem_type] = (++index == 0) ? a : b;
            }
        }
    }

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

    T& operator[](dim3_type idx)
    {
        switch (idx) {
        default:
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        }
    }
    const T& operator[](dim3_type idx) const
    {
        switch (idx) {
        default:
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        }
    }

    iterator<dim3*> max_dim() { return std::max_element(begin(), end()); }
    iterator<const dim3*, const T> max_dim() const { return std::max_element(cbegin(), cend()); }

    iterator<dim3*> begin() { return iterator<dim3*>(this); }
    iterator<dim3*> end() { return iterator<dim3*>(this, true); }

    iterator<const dim3*, const T> cbegin() const { return iterator<const dim3*, const T>(this); }
    iterator<const dim3*, const T> cend() const { return iterator<const dim3*, const T>(this, true); }
    iterator<const dim3*, const T> begin() const { return iterator<const dim3*, const T>(this); }
    iterator<const dim3*, const T> end() const { return iterator<const dim3*, const T>(this, true); }
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