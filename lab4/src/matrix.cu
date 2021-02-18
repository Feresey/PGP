#include "helpers.cuh"

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