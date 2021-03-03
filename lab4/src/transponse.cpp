#include <cstdio>

#include "helpers.cuh"

dev_matrix transponse(const dev_matrix& A, const uint32_t n, const uint32_t m)
{
    fprintf(stderr, "n = %d\n m = %d\n", n, m);
    dev_matrix res;
    res.resize(m * n);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < m; ++j) {
            res[j * n + i] = A[i * m + j];
        }
    }
    return res;
}
