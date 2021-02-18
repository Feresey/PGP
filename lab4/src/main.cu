#include <cmath>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include "helpers.cuh"

// да, костыль
dim3 blocks = 1;
dim3 threads = 32;

typedef thrust::host_vector<double> host_matrix;
typedef thrust::device_vector<double> dev_matrix;

void read_matrix(host_matrix& out, const int n, const int m)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            scanf("%lf", &out[i * n + m]);
        }
    }
}

void show_matrix(FILE* out, const host_matrix& data, const int n, const int m)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fprintf(out, "%e ", data[i * n + m]);
        }
        fprintf(out, "\n");
    }
}

struct comparator {
    __host__ __device__ bool operator()(double a, double b)
    {
        return std::fabs(a) < std::fabs(b);
    }
};

__global__ void swapRows(dev_matrix& data, int n, int a, int b)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int j = idx; j < n; j += offset) {
        auto tmp = data[j * n + b];
        data[j * n + b] = data[j * n + a];
        data[j * n + a] = tmp;
    }
}

// разделить все элементы строки на элемент строки, находящийся на диагонали матрицы
__global__ void divide(dev_matrix& data, int n, int i)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    const double divider = data[i * n + i];

    for (int j = idx + i + 1; j < n; j += offset) {
        data[i * n + j] /= divider;
    }
}

// __global__ void kernel(double *data, int n, int i)
// {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int idy = blockDim.y * blockIdx.y + threadIdx.y;
//     int offsetx = blockDim.x * gridDim.x;
//     int offsety = blockDim.y * gridDim.y;

//     for (int j = idx + i + 1; j < n; j += offsetx) {
//         for (int k = idy + i + 1; k < n; k += offsety) {
//             data[k * n + j] -= data[i * n + j] * data[k * n + i];
//         }
//     }
// }

__global__ void make_joined(dev_matrix& out, const dev_matrix& matrix, const int n, const int m)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < n; i += offsetx) {
        int line = i * 2 * n;
        // главная диагональ смежной матрицы
        out[line + (i + n)] = 1.0;
        for (int j = idy; j < m; j += offsety) {
            out[line + j] = matrix[i * n + j];
        }
    }
}

__global__ void split_joined(dev_matrix& out, const dev_matrix& matrix, const int n, const int m)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < n; i += offsetx) {
        for (int j = idy; j < m; j += offsety) {
            out[n * i + j] = matrix[i * 2 * n + j + n];
        }
    }
}

dev_matrix inverse(const dev_matrix& matrix, const int n, const int m)
{
    dev_matrix joined = dev_matrix(n * 2 * m);
    START_KERNEL((make_joined<<<blocks, threads>>>(joined, matrix, n, m)));

    show_matrix(stdout, joined, n * 2, m);

    double prod = 1;

    comparator comp;
    for (int i = 0; i < n - 1; ++i) {
        // поиск максимального элемента в столбце
        // он гарантированно не нулевой, т.к. матрица не вырожденная
        dev_matrix::iterator iter = joined.begin() + i * n * 2;
        dev_matrix::iterator i_max = thrust::max_element(iter + i, iter + n, comp);

        int mx = iter - i_max;
        if (mx != i) {
            START_KERNEL((swapRows<<<blocks, threads>>>(joined, n * 2, i, mx)));
        }

        divide<<<blocks, threads>>>(matrix, n, i);
        prod /= *i_max;

        // kernel<<<dim3(32, 32), dim3(32, 32)>>>(data, n, i);
    }

    dev_matrix res = dev_matrix(n * m);
    START_KERNEL((split_joined<<<blocks, threads>>>(res, joined, n, m)));

    return res;
}

dev_matrix solve(host_matrix A, host_matrix B, const int n, const int m, const int k)
{
    return {};
}

int main()
{
    int n, m, k;
    std::cin >> n >> m >> k;

    host_matrix A(n * m), B(n * k);

    read_matrix(n, m, A);
    read_matrix(n, k, B);

    host_matrix X = solve(X, A, B, n, m, k);
    show_matrix(stdout, n, k, X);

    return 0;
}