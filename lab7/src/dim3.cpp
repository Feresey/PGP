#include "solver.hpp"

template <class T>
dim3<T>::dim3_print::dim3_print(const dim3<T>& data)
    : data(data)
{
}

template <class T>
std::ostream& operator<<(std::ostream& out, const dim3<T>& data)
{
    out
        << prefix + "_x: " << data.x
        << prefix + "_y: " << data.y
        << prefix + "_z: " << data.z
        << std::endl;
    return out;
}

template <class T>
dim3<T>::dim3(T x, T y, T z)
    : x(x)
    , y(y)
    , z(z)
{
}

template <class T>
dim3<T>::dim3(std::istream& in)
{
    in >> x >> y >> z;
}

template <class T>
std::istream& operator>>(std::istream& in, dim3<T>& val)
{
    val = dim3(in);
    return in;
}

template <class T>
dim3<T>::dim3_print dim3<T>::print(const std::string& prefix)
{
    return dim3_print(this, prefix);
}