#ifndef MAT4_HPP
#define MAT4_HPP

#include <iostream>

#include "helpers.cuh"

class vec3;
class vec4;

struct mat4 {
    float m[4][4];

    __host__ __device__ mat4(
        float m11 = 0.0, float m12 = 0.0, float m13 = 0.0, float m14 = 0.0,
        float m21 = 0.0, float m22 = 0.0, float m23 = 0.0, float m24 = 0.0,
        float m31 = 0.0, float m32 = 0.0, float m33 = 0.0, float m34 = 0.0,
        float m41 = 0.0, float m42 = 0.0, float m43 = 0.0, float m44 = 0.0);

    __host__ __device__ vec3 homogeneous_mult(const vec3& v) const;
    __host__ __device__ friend vec4 operator*(const mat4& m, const vec4& v);

    friend std::ostream& operator<<(std::ostream& os, const mat4& m);
};

#endif
