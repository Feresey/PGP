#include <assert.h>

#include "helpers.cuh"

void read_matrix(host_matrix& out, const size_t n, const size_t m)
{
    size_t junk;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            junk = scanf("%lf", &out[i * m + j]);
        }
    }
    junk = 42;
    assert(junk == 42);
}

void show_matrix(FILE* out, const host_matrix& data, const size_t n, const size_t m)
{
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            fprintf(out, "%.10e ", data[i * m + j]);
        }
        fprintf(out, "\n");
    }
}

#ifdef __cplusplus
void __syncthreads() { }
#endif