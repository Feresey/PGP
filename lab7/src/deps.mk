build/dim3.cpp.o: dim3.cpp dim3.hpp helpers.hpp
dim3.hpp: helpers.hpp
build/grid.cpp.o: dim3.hpp grid.cpp grid.hpp helpers.hpp
grid.hpp: dim3.hpp helpers.hpp
helpers.hpp: 
build/main.cpp.o: helpers.hpp main.cpp dim3.hpp grid.hpp helpers.hpp solver/problem.hpp solver/solver.hpp
build/solver__common.cpp.o: solver/common.cpp dim3.hpp grid.hpp helpers.hpp solver/problem.hpp solver/solver.hpp
build/solver__problem.cpp.o: dim3.hpp grid.hpp helpers.hpp solver/problem.cpp solver/problem.hpp
solver__problem.hpp: dim3.hpp grid.hpp helpers.hpp solver/problem.hpp
build/solver__solver.cpp.o: dim3.hpp grid.hpp helpers.hpp solver/problem.hpp solver/solver.cpp solver/solver.hpp
solver__solver.hpp: dim3.hpp grid.hpp helpers.hpp solver/problem.hpp solver/solver.hpp
