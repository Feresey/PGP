#ifndef TASK_HPP
#define TASK_HPP

#include "vec/vec3.hpp"

#include <vector>

struct Polygon {
    vec3 a;
    vec3 b;
    vec3 c;
    vec3 color;
};

struct Camera {
    float r0, z0, phi0, ar, az, wr, wz, wphi, pr, pz;

    friend std::istream& operator>>(std::istream& is, Camera& p);
    friend std::ostream& operator<<(std::ostream& os, const Camera& p);
};

struct Object {
    vec3 center;
    vec3 color;
    float radius;
    float kreflection, krefraction;
    int nlights;

    friend std::istream& operator>>(std::istream& is, Object& p);
    friend std::ostream& operator<<(std::ostream& os, const Object& p);
};

struct Floor {
    vec3 a, b, c, d;
    std::string texture_path;
    vec3 color;
    float kreflection;

    friend std::istream& operator>>(std::istream& is, Floor& p);
    friend std::ostream& operator<<(std::ostream& os, const Floor& p);

    void mpi_bcast();
};

struct Light {
    vec3 pos;
    vec3 color;

    friend std::istream& operator>>(std::istream& is, Light& p);
    friend std::ostream& operator<<(std::ostream& os, const Light& p);
};

struct Scene {
    int n_frames;
    std::string output_pattern;
    int w, h;
    float angle;
    Camera cum_center, cum_dir;
    Object octahedron, dodecahedron, icosahedron;
    Floor floor;
    int n_lights;
    std::vector<Light> lights;

    friend std::istream& operator>>(std::istream& is, Scene& p);
    friend std::ostream& operator<<(std::ostream& os, const Scene& p);

    void mpi_bcast();
};

#endif
