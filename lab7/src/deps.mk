build/common.cpp.o: common.cpp dim3/dim3.hpp grid/grid.hpp helpers.hpp helpers.hpp problem.hpp solver.hpp
build/dim3__dim3.cpp.o: dim3/dim3.cpp dim3/dim3.hpp helpers.hpp
dim3__dim3.hpp: dim3/dim3.hpp helpers.hpp
build/exchange.cpp.o: dim3/dim3.hpp exchange.cpp exchange.hpp grid/grid.hpp helpers.hpp problem.hpp
exchange.hpp: dim3/dim3.hpp grid/grid.hpp helpers.hpp problem.hpp
build/grid__grid.cpp.o: dim3/dim3.hpp grid/grid.cpp grid/grid.hpp helpers.hpp
grid__grid.hpp: dim3/dim3.hpp grid/grid.hpp helpers.hpp
helpers.hpp: 
build/main.cpp.o: dim3/dim3.hpp grid/grid.hpp helpers.hpp helpers.hpp main.cpp problem.hpp solver.hpp
build/problem.cpp.o: dim3/dim3.hpp grid/grid.hpp helpers.hpp problem.cpp problem.hpp
problem.hpp: dim3/dim3.hpp grid/grid.hpp helpers.hpp
build/solver.cpp.o: dim3/dim3.hpp grid/grid.hpp helpers.hpp helpers.hpp problem.hpp solver.cpp solver.hpp
solver.hpp: dim3/dim3.hpp grid/grid.hpp helpers.hpp problem.hpp
