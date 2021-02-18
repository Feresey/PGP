#include <cmath>
#include <iostream>

#include "helpers.cuh"
#include "solve.cuh"
#include "kernels.cuh"

int main()
{
    int n, m, k;
    std::cin >> n >> m >> k;

    host_matrix A(n * m), B(n * k);

    read_matrix(A, n, m);
    read_matrix(B, n, k);

    host_matrix X = solve(A, B, n, m, k);
    show_matrix(stdout, X, n, k);

    return 0;
}