#include <cmath>
#include <iostream>

#include "helpers.cuh"

int main()
{
    int n, m;
    std::cin >> n >> m;

    host_matrix A(n * m);
    read_matrix(A, n, m);
    host_matrix res = transponse(A, n, m);
    show_matrix(stdout, res, m, n);
    return 0;
}