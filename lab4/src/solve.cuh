#ifndef SOLVE_CUH
#define SOLVE_CUH

dev_matrix inverse(const dev_matrix& matrix, const int n, const int m);
dev_matrix solve(host_matrix A, host_matrix B, const int n, const int m, const int k);

#endif
