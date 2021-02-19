#include "helpers.cuh"
#include "kernels.cuh"

struct comparator {
    __host__ __device__ bool operator()(double a, double b)
    {
        return std::fabs(a) < std::fabs(b);
    }
};

dev_matrix solve(const dev_matrix& A, const dev_matrix& B, const int n, const int m, const int k)
{
    const double* B_raw = thrust::raw_pointer_cast(&B[0]);
    dev_matrix B_trans(k * n);
    double* B_trans_raw = thrust::raw_pointer_cast(&B_trans[0]);
    START_KERNEL((transponse<<<BLOCKS, THREADS>>>(B_trans_raw, B_raw, n, k)));

    show_matrix(stderr, B, n, k);
    show_matrix(stderr, B_trans, k, n);

    dev_matrix res(m * k);
    return res;
}