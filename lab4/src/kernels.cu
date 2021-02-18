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

// __global__ void kernel(double *data, int n, int i)
// {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int idy = blockDim.y * blockIdx.y + threadIdx.y;
//     int offsetx = blockDim.x * gridDim.x;
//     int offsety = blockDim.y * gridDim.y;

//     for (int j = idx + i + 1; j < n; j += offsetx) {
//         for (int k = idy + i + 1; k < n; k += offsety) {
//             data[k * n + j] -= data[i * n + j] * data[k * n + i];
//         }
//     }
// }

__global__ void make_joined(double* out, const double* matrix, const int n, const int m)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < n; i += offsetx) {
        int line = i * 2 * n;
        // главная диагональ смежной матрицы
        out[line + (i + n)] = 1.0;
        for (int j = idy; j < m; j += offsety) {
            out[line + j] = matrix[i * n + j];
        }
    }
}

__global__ void split_joined(double* out, const double* matrix, const int n, const int m)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < n; i += offsetx) {
        for (int j = idy; j < m; j += offsety) {
            out[n * i + j] = matrix[i * 2 * n + j + n];
        }
    }
}