#include <iostream>

#include "helpers.cuh"

int main(){
    size_t n, m, k;
    std::cin >> n >> m >> k;

    host_matrix A(n * m), B(n * k);

    read_matrix(A, n, m);
    read_matrix(B, n, k);

    host_matrix X = solve(dev_matrix(A), dev_matrix(B), n, m, k);
    show_matrix(stdout, X, m, k);

    return 0;
}