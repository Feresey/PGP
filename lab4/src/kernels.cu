__global__ void swapRows(double* data, int n, int a, int b)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int j = idx; j < n; j += offset) {
        double tmp = data[j * n + b];
        data[j * n + b] = data[j * n + a];
        data[j * n + a] = tmp;
    }
}

// разделить все элементы строки на элемент строки, находящийся на диагонали матрицы
__global__ void divide(double* data, int n, int i)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    const double divider = data[i * n + i];

    for (int j = idx + i + 1; j < n; j += offset) {
        data[i * n + j] /= divider;
    }
}

#define BLOCK_SIZE 32

__global__ void transponse(double* out, const double* data, int n, int m)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    __shared__ double shared[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (int i = idx; i < n; i += offsetx) {
        temp[idx] = ;
        for (int j = idy; j < m; j += offsety) {
            out[j * n + i] = data[i * m + j];

            __syncthreads();
        }
    }
}