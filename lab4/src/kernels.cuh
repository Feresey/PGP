#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void swapRows(double* data, int n, int a, int b);
// разделить все элементы строки на элемент строки, находящийся на диагонали матрицы
__global__ void divide(double* data, int n, int i);
// __global__ void kernel(double *data, int n, int i);

__global__ void transponse(double* out, const double* data, int n, int m);
#endif