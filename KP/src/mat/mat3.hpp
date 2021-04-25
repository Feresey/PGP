#ifndef MAT3_HPP
#define MAT3_HPP

#include <iostream>

#include "helpers.cuh"

class vec3;
class vec4;

struct mat3 {
    float m[3][3];

    HD mat3(
        float m11 = 0.0, float m12 = 0.0, float m13 = 0.0,
        float m21 = 0.0, float m22 = 0.0, float m23 = 0.0,
        float m31 = 0.0, float m32 = 0.0, float m33 = 0.0);

    friend std::ostream& operator<<(std::ostream& os, const mat3& m);

    HD friend mat3 operator*(const mat3& a, const mat3& b);
    HD friend mat3 operator+(const mat3& a, const mat3& b);
    HD friend mat3 operator*(float a, const mat3& m);
    HD friend mat3 operator*(const mat3& m, float a);
    HD friend vec3 operator*(const mat3& m, const vec3& v);

    HD float det() const;
    HD mat3 inverse() const;
    HD static mat3 identity();
    HD static mat3 align_mat(const vec3& a, const vec3& b);
};

#endif
