#include "vec3.hpp"

HD vec3::vec3()
    : x(0.0)
    , y(0.0)
    , z(0.0)
{
}
HD vec3::vec3(float val)
    : x(val)
    , y(val)
    , z(val)
{
}
HD vec3::vec3(float x, float y, float z)
    : x(x)
    , y(y)
    , z(z)
{
}

std::ostream& operator<<(std::ostream& os, const vec3& f)
{
    os << f.x << ' ' << f.y << ' ' << f.z;
    return os;
}

std::istream& operator>>(std::istream& is, vec3& f)
{
    is >> f.x >> f.y >> f.z;
    return is;
}

HD float vec3::len() const { return sqrt(x * x + y * y + z * z); }

HD vec3 vec3::normalize() const
{
    float l = len();
    return { x / l, y / l, z / l };
}

HD float vec3::dot_product(const vec3& a, const vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HD vec3 vec3::cross_product(const vec3& a, const vec3& b)
{
    vec3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;

    return res;
}

HD vec3 vec3::reflect(const vec3 vec,  const vec3 normal)
{
    return vec - 2.0f * vec3::dot_product(vec, normal) * normal;
}
HD vec3 vec3::refract(const vec3 vec, const vec3& normal, float n1, float n2)
{
    float r = n1 / n2;
    float c = -vec3::dot_product(normal, vec);
    return r *vec + (r * c - sqrt(1.0f - r * r * (1.0f - c * c))) * normal;
}

HD vec3 operator+(const vec3& a, const vec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
HD vec3 operator-(const vec3& a) { return { -a.x, -a.y, -a.z }; }
HD vec3 operator-(const vec3& a, const vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
HD vec3 operator*(float c, const vec3& v) { return { c * v.x, c * v.y, c * v.z }; }
HD vec3 operator*(const vec3& v, float c) { return c * v; }
HD vec3 operator*(const vec3& v, const vec3& w) { return { v.x * w.x, v.y * w.y, v.z * w.z }; }

HD void operator+=(vec3& a, const vec3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

HD void operator-=(vec3& a, const vec3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
