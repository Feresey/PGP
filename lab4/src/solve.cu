#include "helpers.cuh"
#include "kernels.cuh"

struct comparator {
    __host__ __device__ bool operator()(double a, double b)
    {
        return std::fabs(a) < std::fabs(b);
    }
};

dev_matrix inverse(const dev_matrix& matrix, const int n, const int m)
{
    dev_matrix joined = dev_matrix(n * 2 * m);
    double* joined_raw = thrust::raw_pointer_cast(&joined[0]);
    const double* matrix_raw = thrust::raw_pointer_cast(&matrix[0]);
    START_KERNEL((make_joined<<<BLOCKS, THREADS>>>(joined_raw, matrix_raw, n, m)));

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
            START_KERNEL((swapRows<<<BLOCKS, THREADS>>>(joined_raw, n * 2, i, mx)));
        }

        START_KERNEL((divide<<<BLOCKS, THREADS>>>(joined_raw, n, i)));
        prod /= *i_max;

        // kernel<<<dim3(32, 32), dim3(32, 32)>>>(data, n, i);
    }

    dev_matrix res = dev_matrix(n * m);
    double* res_raw = thrust::raw_pointer_cast(&res[0]);
    START_KERNEL((split_joined<<<BLOCKS, THREADS>>>(res_raw, joined_raw, n, m)));

    return res;
}

dev_matrix solve(host_matrix A, host_matrix B, const int n, const int m, const int k)
{
    return inverse(A, n, m);
}