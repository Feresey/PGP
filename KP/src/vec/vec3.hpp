#ifndef VEC3_HPP
#define VEC3_HPP

#include <cmath>
#include <iostream>

#include "helpers.cuh"

struct vec3 {
    float x, y, z;

    HD vec3();
    HD vec3(float val);
    HD vec3(float x, float y, float z);

    HD float len() const;
    HD vec3 normalize() const;
    HD static vec3 cross_product(const vec3& a, const vec3& b);
    HD static float dot_product(const vec3& a, const vec3& b);
    HD static vec3 reflect(const vec3 vec, const vec3 normal);
    HD static vec3 refract(const vec3 vec, const vec3& normal, float n1, float n2);

    friend std::ostream& operator<<(std::ostream& os, const vec3& f);
    friend std::istream& operator>>(std::istream& is, vec3& f);

    HD friend vec3 operator+(const vec3& a, const vec3& b);
    HD friend void operator+=(vec3& a, const vec3& b);
    HD friend vec3 operator-(const vec3& a);
    HD friend vec3 operator-(const vec3& a, const vec3& b);
    HD friend void operator-=(vec3& a, const vec3& b);
    HD friend vec3 operator*(float c, const vec3& v);
    HD friend vec3 operator*(const vec3& v, float c);
    HD friend vec3 operator*(const vec3& v, const vec3& w);
} __attribute__((aligned(16)));

#endif
