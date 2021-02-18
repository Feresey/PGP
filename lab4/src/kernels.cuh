#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void swapRows(double* data, int n, int a, int b);
// разделить все элементы строки на элемент строки, находящийся на диагонали матрицы
__global__ void divide(double* data, int n, int i);
// __global__ void kernel(double *data, int n, int i);
__global__ void make_joined(double* out, const double* matrix, const int n, const int m);
__global__ void split_joined(double* out, const double* matrix, const int n, const int m);

#endif