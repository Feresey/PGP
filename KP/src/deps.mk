build/ALL.cu.o: ALL.cu helpers.cuh helpers.hpp main.cpp render/objects.cpp render/render.cpp render/render.cuh scene.cpp scene.hpp ssaa.hpp vec/vec3.hpp
build/helpers.cu.o: helpers.cu helpers.cuh
build/mat__mat.cu.o: helpers.cuh mat/mat3.cpp mat/mat3.hpp mat/mat4.cpp mat/mat4.hpp mat/mat.cu vec/vec3.hpp vec/vec4.hpp
build/render__render.cu.o: helpers.cuh helpers.hpp mat/mat3.hpp render/render.cu render/render.cuh scene.hpp ssaa.hpp vec/vec3.hpp
build/ssaa.cu.o: helpers.cuh ssaa.cu ssaa.hpp vec/vec3.hpp
build/vec__vec.cu.o: helpers.cuh vec/vec3.cpp vec/vec3.hpp vec/vec4.cpp vec/vec4.hpp vec/vec.cu
