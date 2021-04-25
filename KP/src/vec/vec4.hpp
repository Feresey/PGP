#ifndef VEC4_HPP
#define VEC4_HPP

#include "helpers.cuh"

#include <cmath>

struct vec4 {
    float x, y, z, w;

    HD vec4();
    HD vec4(float val);
    HD vec4(float x, float y, float z, float w);

    friend std::ostream& operator<<(std::ostream& os, const vec4& f);
    friend std::istream& operator>>(std::istream& is, vec4& f);

    HD float len() const;
    HD vec4 normalize() const;

    HD friend vec4 operator+(const vec4& a, const vec4& b);
    HD friend void operator+=(vec4& a, const vec4& b);
    HD friend vec4 operator-(const vec4& a);
    HD friend vec4 operator-(const vec4& a, const vec4& b);
    HD friend void operator-=(vec4& a, const vec4& b);
    HD friend vec4 operator*(float c, const vec4& v);
    HD friend vec4 operator*(const vec4& v, const vec4& w);
    HD friend vec4 operator*(const vec4& v, float c);
} __attribute__((aligned(16)));

#endif
