build/dim3__dim3.cpp.o: dim3/dim3.cpp dim3/dim3.hpp helpers.hpp
build/exchange.cpp.o: dim3/dim3.hpp exchange.cpp exchange.hpp grid/grid.hpp helpers.cuh helpers.hpp pool/kernels.hpp pool/pool.hpp pool/task.hpp
build/grid__grid.cu.o: dim3/dim3.hpp grid/grid.cpp grid/grid.cu grid/grid.hpp helpers.cuh helpers.hpp
build/main.cpp.o: dim3/dim3.hpp grid/grid.hpp helpers.cuh helpers.hpp main.cpp pool/kernels.hpp pool/pool.hpp pool/task.hpp solver.hpp
build/pool__kernels.cu.o: dim3/dim3.hpp grid/grid.hpp helpers.cuh helpers.hpp pool/kernels.cu pool/kernels.hpp
build/pool__pool.cpp.o: dim3/dim3.hpp grid/grid.hpp helpers.cuh helpers.hpp pool/kernels.hpp pool/pool.cpp pool/pool.hpp pool/task.hpp
build/pool__pool.cu.o: dim3/dim3.hpp grid/grid.hpp helpers.cuh helpers.hpp pool/kernels.hpp pool/pool.cu pool/pool.hpp pool/task.hpp
build/pool__task.cpp.o: dim3/dim3.hpp helpers.hpp pool/task.cpp pool/task.hpp
build/solver.cpp.o: dim3/dim3.hpp exchange.hpp grid/grid.hpp helpers.cuh helpers.hpp pool/kernels.hpp pool/pool.hpp pool/task.hpp solver.cpp solver.hpp
