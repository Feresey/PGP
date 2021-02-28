#include <cmath>
#include <iostream>

#include "helpers.cuh"

int main()
{
    int n, m;
    std::cin >> n >> m;

    if (n == 0 || m == 0) {
        return 0;
    }

    host_matrix A(n * m);
    read_matrix(A, n, m);
    host_matrix res = transponse(A, n, m);
    show_matrix(stdout, res, m, n);
    return 0;
}