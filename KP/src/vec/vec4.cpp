#include "vec4.hpp"

HD vec4::vec4()
    : x(0.0f)
    , y(0.0f)
    , z(0.0f)
    , w(0.0f)
{
}
HD vec4::vec4(float val)
    : x(val)
    , y(val)
    , z(val)
    , w(val)
{
}
HD vec4::vec4(float x, float y, float z, float w)
    : x(x)
    , y(y)
    , z(z)
    , w(w)
{
}

std::ostream& operator<<(std::ostream& os, const vec4& f)
{
    os << f.x << ' ' << f.y << ' ' << f.z << ' ' << f.w;
    return os;
}

std::istream& operator>>(std::istream& is, vec4& f)
{
    is >> f.x >> f.y >> f.z >> f.w;
    return is;
}

HD float vec4::len() const { return std::sqrt(x * x + y * y + z * z + w * w); }

HD vec4 vec4::normalize() const
{
    float l = len();
    return { x / l, y / l, z / l, w / l };
}

HD vec4 operator+(const vec4& a, const vec4& b) { return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
HD vec4 operator-(const vec4& a) { return { -a.x, -a.y, -a.z, -a.w }; }
HD vec4 operator-(const vec4& a, const vec4& b) { return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
HD vec4 operator*(float c, const vec4& v) { return { c * v.x, c * v.y, c * v.z, c * v.w }; }
HD vec4 operator*(const vec4& v, const vec4& w) { return { v.x * w.x, v.y * w.y, v.z * w.z, v.w * w.w }; }
HD vec4 operator*(const vec4& v, float c) { return c * v; }

HD void operator+=(vec4& a, const vec4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

HD void operator-=(vec4& a, const vec4& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
