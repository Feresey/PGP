#include "mat4.hpp"

#include "vec/vec3.hpp"
#include "vec/vec4.hpp"

std::ostream& operator<<(std::ostream& os, const mat4& m)
{
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            os << m.m[i][j] << ' ';
        }
        os << std::endl;
    }
    return os;
}

__host__ __device__ mat4::mat4(
    float m11, float m12, float m13, float m14,
    float m21, float m22, float m23, float m24,
    float m31, float m32, float m33, float m34,
    float m41, float m42, float m43, float m44)
{
    m[0][0] = m11;
    m[0][1] = m12;
    m[0][2] = m13;
    m[0][3] = m14;
    m[1][0] = m21;
    m[1][1] = m22;
    m[1][2] = m23;
    m[1][3] = m24;
    m[2][0] = m31;
    m[2][1] = m32;
    m[2][2] = m33;
    m[2][3] = m34;
    m[3][0] = m41;
    m[3][1] = m42;
    m[3][2] = m43;
    m[3][3] = m44;
}

__host__ __device__ vec4 operator*(const mat4& m, const vec4& v)
{
    vec4 res;
    res.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w;
    res.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w;
    res.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w;
    res.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w;
    return res;
}

__host__ __device__ vec3 mat4::homogeneous_mult(const vec3& v) const
{
    vec4 tmp(v.x, v.y, v.z, 1.0f);
    tmp = (*this) * tmp;
    return { tmp.x, tmp.y, tmp.z };
}
