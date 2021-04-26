#include <iostream>

#include "vec/vec3.hpp"

#include "mat3.hpp"

std::ostream& operator<<(std::ostream& os, const mat3& m)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            os << m.m[i][j] << ' ';
        }
        os << std::endl;
    }
    return os;
}

HD mat3::mat3(
    float m11, float m12, float m13,
    float m21, float m22, float m23,
    float m31, float m32, float m33)
{
    m[0][0] = m11;
    m[0][1] = m12;
    m[0][2] = m13;
    m[1][0] = m21;
    m[1][1] = m22;
    m[1][2] = m23;
    m[2][0] = m31;
    m[2][1] = m32;
    m[2][2] = m33;
}

HD mat3 mat3::identity()
{
    mat3 res;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            res.m[i][j] = 0.0f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        res.m[i][i] = 1.0f;
    }
    return res;
}

HD float mat3::det() const
{
    return 0.0
        + m[0][0] * m[1][1] * m[2][2]
        + m[1][0] * m[0][2] * m[2][1]
        + m[2][0] * m[0][1] * m[1][2]

        - m[0][2] * m[1][1] * m[2][0]
        - m[0][0] * m[1][2] * m[2][1]
        - m[0][1] * m[1][0] * m[2][2];
}

HD mat3 mat3::inverse() const
{
    float d = det();

    float m11 = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) / d;
    float m12 = (m[2][1] * m[0][2] - m[0][1] * m[2][2]) / d;
    float m13 = (m[0][1] * m[1][2] - m[1][1] * m[0][2]) / d;

    float m21 = (m[2][0] * m[1][2] - m[1][0] * m[2][2]) / d;
    float m22 = (m[0][0] * m[2][2] - m[2][0] * m[0][2]) / d;
    float m23 = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) / d;

    float m31 = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) / d;
    float m32 = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) / d;
    float m33 = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) / d;

    return mat3(
        m11, m12, m13,
        m21, m22, m23,
        m31, m32, m33);
}

HD mat3 mat3::align_mat(const vec3& a, const vec3& b)
{
    vec3 v = vec3::cross_product(a, b);
    float c = vec3::dot_product(a, b);

    mat3 m(
        0.0f, -v.z, v.y,
        v.z, 0.0f, -v.x,
        -v.y, v.x, 0.0f);

    return mat3::identity() + m + 1.0f / (1.0f + c) * m * m;
}

HD mat3 operator*(const mat3& a, const mat3& b)
{
    mat3 res;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum += a.m[i][k] * b.m[k][j];
            }
            res.m[i][j] = sum;
        }
    }
    return res;
}

HD mat3 operator+(const mat3& a, const mat3& b)
{
    mat3 res;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            res.m[i][j] = a.m[i][j] + b.m[i][j];
        }
    }

    return res;
}

HD mat3 operator*(float a, const mat3& m)
{
    mat3 res;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            res.m[i][j] = a * m.m[i][j];
        }
    }
    return res;
}

HD mat3 operator*(const mat3& m, float a)
{
    return a * m;
}

HD vec3 operator*(const mat3& m, const vec3& v)
{
    vec3 res;
    res.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z;
    res.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z;
    res.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z;
    return res;
}
