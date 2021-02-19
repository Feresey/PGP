#ifndef SOLVE_CUH
#define SOLVE_CUH

dev_matrix inverse(const dev_matrix& matrix, const int n, const int m);
dev_matrix solve(const dev_matrix& A, const dev_matrix& B, const int n, const int m, const int k);

#endif
