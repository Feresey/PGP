#include <mpi.h>
#include <typeinfo>

#include "helpers.hpp"
#include "scene.hpp"

std::istream& operator>>(std::istream& is, Camera& p)
{
    is >> p.r0 >> p.z0 >> p.phi0 >> p.ar >> p.az >> p.wr >> p.wz >> p.wphi >> p.pr >> p.pz;
    return is;
}
std::ostream& operator<<(std::ostream& os, const Camera& p)
{
    os << p.r0 << ' ' << p.z0 << ' ' << p.phi0 << ' ' << p.ar << ' ' << p.az
       << ' ' << p.wr << ' ' << p.wz << ' ' << p.wphi << ' ' << p.pr << ' '
       << p.pz;
    return os;
}

std::istream& operator>>(std::istream& is, Object& p)
{
    is >> p.center >> p.color >> p.radius >> p.kreflection >> p.krefraction >> p.nlights;
    return is;
}
std::ostream& operator<<(std::ostream& os, const Object& p)
{
    os << p.center << ' ' << p.color << ' ' << p.radius << ' '
       << p.kreflection << ' ' << p.krefraction << ' ' << p.nlights;
    return os;
}

std::istream& operator>>(std::istream& is, Floor& p)
{
    is >> p.a >> p.b >> p.c >> p.d >> p.texture_path >> p.color >> p.kreflection;
    return is;
}
std::ostream& operator<<(std::ostream& os, const Floor& p)
{
    os << p.a << ' ' << p.b << ' ' << p.c << ' ' << p.d << ' '
       << p.texture_path << ' ' << p.color << ' ' << p.kreflection;
    return os;
}

std::istream& operator>>(std::istream& is, Light& p)
{
    is >> p.pos >> p.color;
    return is;
}
std::ostream& operator<<(std::ostream& os, const Light& p)
{
    os << p.pos << ' ' << p.color;
    return os;
}

std::istream& operator>>(std::istream& is, Scene& p)
{
    is
        >> p.n_frames
        >> p.output_pattern
        >> p.w >> p.h >> p.angle
        >> p.cum_center >> p.cum_dir
        >> p.octahedron >> p.dodecahedron >> p.icosahedron
        >> p.floor
        >> p.n_lights;
    p.lights.resize(size_t(p.n_lights));
    for (int i = 0; i < p.n_lights; ++i) {
        is >> p.lights[size_t(i)];
    }
    return is;
}
std::ostream& operator<<(std::ostream& os, const Scene& p)
{
    os << p.n_frames << ' ' << p.output_pattern << '\n'
       << p.w << ' ' << p.h << ' ' << p.angle << '\n'
       << "camera center: " << p.cum_center << '\n'
       << "camera dir: " << p.cum_dir << '\n'
       << "octa: " << p.octahedron << '\n'
       << "doda: " << p.dodecahedron << '\n'
       << "icos: " << p.icosahedron << '\n'
       << "floor: " << p.floor << '\n'
       << "nlights: " << p.lights.size() << '\n';
    for (auto& it : p.lights)
        os << it;
    return os;
}

void Floor::mpi_bcast()
{
    bcast_vec3(&a);
    bcast_vec3(&b);
    bcast_vec3(&c);
    bcast_vec3(&d);
    bcast_vec3(&color);
    bcast_int(&kreflection);

    int texture_path_size = int(texture_path.size());
    bcast_int(&texture_path_size);
    texture_path.resize(size_t(texture_path_size));
    bcast_bytes((void*)texture_path.data(), texture_path_size);
}

void Scene::mpi_bcast()
{
    bcast_int(&n_frames);
    bcast_int(&w);
    bcast_int(&h);
    bcast_float(&angle);
    bcast_bytes(&cum_center, sizeof(Camera));
    bcast_bytes(&cum_dir, sizeof(Camera));
    floor.mpi_bcast();

    bcast_bytes(&octahedron, sizeof(Object));
    bcast_bytes(&dodecahedron, sizeof(Object));
    bcast_bytes(&icosahedron, sizeof(Object));

    int output_pattern_size = int(output_pattern.size());
    bcast_int(&output_pattern_size);
    output_pattern.resize(size_t(output_pattern_size));
    bcast_bytes((void*)output_pattern.data(), output_pattern_size);

    bcast_int(&n_lights);
    lights.resize(size_t(n_lights));
    bcast_bytes(lights.data(), int(sizeof(Light)) * n_lights);
}
