all:
	nvcc -G -arch=sm_35 --std=c++11 -Werror cross-execution-space-call -lm main.cu -o a.out --compiler-options "-Wall -Werror -Wextra -g3"