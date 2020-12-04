all:
	nvcc \
		--std=c++14 -Werror cross-execution-space-call \
		-arch=sm_35 -Wno-deprecated-gpu-targets\
		-G \
		--compiler-options "-Wall -Werror -Wextra -g3 -Wconversion -Wsign-conversion -Wunreachable-code" \
		main.cu -o a.out \
		-lm