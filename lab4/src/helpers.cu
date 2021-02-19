#include "helpers.cuh"

void read_matrix(host_matrix& out, const size_t n, const size_t m)
{
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            scanf("%lf", &out[i * m + j]);
        }
    }
}

void show_matrix(FILE* out, const host_matrix& data, const size_t n, const size_t m)
{
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            fprintf(out, "%e ", data[i * m + j]);
        }
        fprintf(out, "\n");
    }
}