all:
	nvcc --std=c++11 -Werror cross-execution-space-call -lm main.cu -o main
debug:
	gcc -g3 -O0 main.c serialize.c