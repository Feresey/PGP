all:
	nvcc --std=c++11 -Werror cross-execution-space-call -lm main.cu -o main
debug:
	nvcc --std=c++11 -Werror cross-execution-space-call -lm -DBENCHMARK main.cu -o main
	gcc --std=c11 -Werror -lm -o main_cpu main.c