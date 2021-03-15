#include <cmath>

#include "dim3/dim3.hpp"
#include "grid/grid.hpp"
#include "helpers.cuh"

#define BORDER_OPERATIONS_GRID_DIM 32
#define BORDER_OPERATIONS_BLOCK_DIM 32

enum side_tag {
    LEFT,
    RIGHT,
    FRONT,
    BACK,
    TOP,
    BOTTOM
};

enum border_tag {
    UP_DOWN,
    LEFT_RIGHT,
    FRONT_BACK
};

__global__ void get_border(double* out, double* buf, Grid grid, int a_size, int b_size, int border_idx, border_tag tag)
{
    const int id_a = threadIdx.x + blockIdx.x * blockDim.x,
              id_b = threadIdx.y + blockIdx.y * blockDim.y,
              offset_a = blockDim.x * gridDim.x,
              offset_b = blockDim.y * gridDim.y;

    double temp;

    for (int b = id_b; b < b_size; b += offset_b) {
        for (int a = id_a; a < a_size; a += offset_a) {
            switch (tag) {
            case LEFT_RIGHT:
                temp = buf[grid.cell_absolute_id(border_idx, a, b)];
                break;
            case FRONT_BACK:
                temp = buf[grid.cell_absolute_id(a, border_idx, b)];
                break;
            case UP_DOWN:
                temp = buf[grid.cell_absolute_id(a, b, border_idx)];
                break;
            }

            out[b * a_size + a] = temp;
        }
    }
}

__global__ void set_border(double* out, double* buf, Grid grid, int a_size, int b_size, int border_idx, border_tag tag)
{
    const int id_a = threadIdx.x + blockIdx.x * blockDim.x,
              id_b = threadIdx.y + blockIdx.y * blockDim.y,
              offset_a = blockDim.x * gridDim.x,
              offset_b = blockDim.y * gridDim.y;

    int dest_idx;

    for (int b = id_b; b < b_size; b += offset_b) {
        for (int a = id_a; a < a_size; a += offset_a) {
            switch (tag) {
            case LEFT_RIGHT:
                dest_idx = grid.cell_absolute_id(border_idx, a, b);
                break;
            case FRONT_BACK:
                dest_idx = grid.cell_absolute_id(a, border_idx, b);
                break;
            case UP_DOWN:
                dest_idx = grid.cell_absolute_id(a, b, border_idx);
                break;
            }

            out[dest_idx] = buf[b * a_size + a];
        }
    }
}

void horizontal_stack(double* dst, double** buffers, int* horizontal_sizes, int vertical_size, int device_count)
{
    int total_horizontal_size = 0;
    for (int d = 0; d < device_count; ++d)
        total_horizontal_size += horizontal_sizes[d];

    for (int d = 0; d < device_count; ++d) {
        for (int j = 0; j < vertical_size; ++j) {
            for (int i = 0; i < horizontal_sizes[d]; ++i) {
                dst[total_horizontal_size * j + d * horizontal_sizes[0] + i] = buffers[d][j * horizontal_sizes[d] + i];
            }
        }
    }
}

void vertical_stack(double* dst, double** buffers, int horizontal_size, int* vertical_sizes, int device_count)
{
    int offset = 0;
    for (int d = 0; d < device_count; ++d) {
        for (int j = 0; j < vertical_sizes[d]; ++j) {
            for (int i = 0; i < horizontal_size; ++i) {
                dst[offset + j * horizontal_size + i] = buffers[d][j * horizontal_size + i];
            }
        }
        offset += vertical_sizes[d] * horizontal_size;
    }
}

void horizontal_unstack(double* src, double** buffers, int* horizontal_sizes, int vertical_size, int device_count)
{
    int total_horizontal_size = 0;
    for (int d = 0; d < device_count; ++d)
        total_horizontal_size += horizontal_sizes[d];

    for (int d = 0; d < device_count; ++d) {
        for (int j = 0; j < vertical_size; ++j) {
            for (int i = 0; i < horizontal_sizes[d]; ++i) {
                buffers[d][j * horizontal_sizes[d] + i] = src[total_horizontal_size * j + d * horizontal_sizes[0] + i];
            }
        }
    }
}

void vertical_unstack(double* src, double** buffers, int horizontal_size, int* vertical_sizes, int device_count)
{
    int offset = 0;
    for (int d = 0; d < device_count; ++d) {
        for (int j = 0; j < vertical_sizes[d]; ++j) {
            for (int i = 0; i < horizontal_size; ++i) {
                buffers[d][j * horizontal_size + i] = src[offset + j * horizontal_size + i];
            }
        }
        offset += vertical_sizes[d] * horizontal_size;
    }
}

void get_border_multigpu(double* dst, double* dev_dst, double** src, double** buffers, double** dev_buffers,
    int border, int* block_sizes_x, int* block_sizes_y, int* block_sizes_z, int device_count, int split_type)
{
    dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
    dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

    if (split_type == UP_DOWN) {
        if (border == TOP || border == BOTTOM) {
            int src_index = border == BOTTOM ? 0 : device_count - 1;
            int border_index = border == BOTTOM ? 0 : block_sizes_z[device_count - 1] - 1;
            CUDA_ERR(cudaSetDevice(src_index));
            START_KERNEL((get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index)));
            CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyDeviceToHost));
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                START_KERNEL((get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
                CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            vertical_stack(dst, buffers, block_sizes_y[0], block_sizes_z, device_count);
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                START_KERNEL((get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
                CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_z, device_count);
        }
    } else if (split_type == LEFT_RIGHT) {
        if (border == TOP || border == BOTTOM) {
            int border_index = border == BOTTOM ? 0 : block_sizes_z[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                START_KERNEL((get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
                CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
            }
            horizontal_stack(dst, buffers, block_sizes_x, block_sizes_y[0], device_count);
        } else if (border == LEFT || border == RIGHT) {
            int src_index = border == LEFT ? 0 : device_count - 1;
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
            CUDA_ERR(cudaSetDevice(src_index));
            START_KERNEL((get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index)));
            CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                START_KERNEL((get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
                CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            horizontal_stack(dst, buffers, block_sizes_x, block_sizes_z[0], device_count);
        }
    } else if (split_type == FRONT_BACK) {
        if (border == TOP || border == BOTTOM) {
            int border_index = border == BOTTOM ? 0 : block_sizes_z[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                START_KERNEL((get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
                CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
            }
            vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_y, device_count);
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                START_KERNEL((get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
                CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            horizontal_stack(dst, buffers, block_sizes_y, block_sizes_z[0], device_count);
        } else if (border == FRONT || border == BACK) {
            int src_index = border == FRONT ? 0 : device_count - 1;
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
            CUDA_ERR(cudaSetDevice(src_index));
            START_KERNEL((get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index)));
            CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
        }
    } else {
        printf("ERROR OCCURED\n");
        exit(0);
    }
}

void set_border_multigpu(double** dst, double* src, double* dev_src, double** buffers, double** dev_buffers,
    int border, int* block_sizes_x, int* block_sizes_y, int* block_sizes_z, int device_count, int split_type)
{
    dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
    dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

    if (split_type == UP_DOWN) {
        if (border == TOP || border == BOTTOM) {
            int src_index = border == BOTTOM ? 0 : device_count - 1;
            int border_index = border == BOTTOM ? -1 : block_sizes_z[device_count - 1];
            CUDA_ERR(cudaSetDevice(src_index));
            CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyHostToDevice));
            START_KERNEL((set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index)));
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
            vertical_unstack(src, buffers, block_sizes_y[0], block_sizes_z, device_count);
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                START_KERNEL((set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
            }
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
            vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_z, device_count);
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                START_KERNEL((set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
            }
        }
    } else if (split_type == LEFT_RIGHT) {
        if (border == TOP || border == BOTTOM) {
            int border_index = border == BOTTOM ? -1 : block_sizes_z[device_count - 1];
            horizontal_unstack(src, buffers, block_sizes_x, block_sizes_y[0], device_count);
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
                START_KERNEL((set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
            }
        } else if (border == LEFT || border == RIGHT) {
            int src_index = border == LEFT ? 0 : device_count - 1;
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
            CUDA_ERR(cudaSetDevice(src_index));
            CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
            START_KERNEL((set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index)));
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
            horizontal_unstack(src, buffers, block_sizes_x, block_sizes_z[0], device_count);
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                START_KERNEL((set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
            }
        }
    } else if (split_type == FRONT_BACK) {
        if (border == TOP || border == BOTTOM) {
            int border_index = border == BOTTOM ? -1 : block_sizes_z[device_count - 1];
            vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_y, device_count);
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
                START_KERNEL((set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
            }
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
            horizontal_unstack(src, buffers, block_sizes_y, block_sizes_z[0], device_count);
            for (int d = 0; d < device_count; ++d) {
                CUDA_ERR(cudaSetDevice(d));
                CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                START_KERNEL((set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index)));
            }
        } else if (border == FRONT || border == BACK) {
            int src_index = border == FRONT ? 0 : device_count - 1;
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
            CUDA_ERR(cudaSetDevice(src_index));
            CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
            START_KERNEL((set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index)));
        }
    } else {
        printf("ERROR OCCURED\n");
        exit(0);
    }
}

int get_last_index(int split_type, int* block_sizes_x, int* block_sizes_y, int* block_sizes_z, int device)
{
    int last_index;
    if (split_type == UP_DOWN)
        last_index = block_sizes_z[device] - 1;
    if (split_type == LEFT_RIGHT)
        last_index = block_sizes_x[device] - 1;
    if (split_type == FRONT_BACK)
        last_index = block_sizes_y[device] - 1;
    return last_index;
}

void get_intergpu_border(double* dst, double* src, double* dev_buffer, int index, mydim3<int> bsize, int split_type)
{
    dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
    dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

    if (split_type == UP_DOWN) {
        START_KERNEL((get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, bsize.x, bsize.y, bsize.z, index)));
        CUDA_ERR(cudaMemcpy(dst, dev_buffer, sizeof(double) * bsize.x * bsize.y, cudaMemcpyDeviceToHost));
    } else if (split_type == LEFT_RIGHT) {
        START_KERNEL((get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, bsize.x, bsize.y, bsize.z, index)));
        CUDA_ERR(cudaMemcpy(dst, dev_buffer, sizeof(double) * bsize.y * bsize.z, cudaMemcpyDeviceToHost));
    } else if (split_type == FRONT_BACK) {
        START_KERNEL((get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, bsize.x, bsize.y, bsize.z, index)));
        CUDA_ERR(cudaMemcpy(dst, dev_buffer, sizeof(double) * bsize.x * bsize.z, cudaMemcpyDeviceToHost));
    } else {
        printf("ERROR OCCURED\n");
        exit(0);
    }
}

void set_intergpu_border(double* dst, double* src, double* dev_buffer, int index, mydim3<int> bsize, int split_type)
{
    dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
    dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

    if (split_type == UP_DOWN) {
        CUDA_ERR(cudaMemcpy(dev_buffer, src, sizeof(double) * bsize.x * bsize.y, cudaMemcpyHostToDevice));
        START_KERNEL((set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, bsize.x, bsize.y, bsize.z, index)));
    } else if (split_type == LEFT_RIGHT) {
        CUDA_ERR(cudaMemcpy(dev_buffer, src, sizeof(double) * bsize.y * bsize.z, cudaMemcpyHostToDevice));
        START_KERNEL((set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, bsize.x, bsize.y, bsize.z, index)));
    } else if (split_type == FRONT_BACK) {
        CUDA_ERR(cudaMemcpy(dev_buffer, src, sizeof(double) * bsize.x * bsize.z, cudaMemcpyHostToDevice));
        START_KERNEL((set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, bsize.x, bsize.y, bsize.z, index)));
    } else {
        printf("ERROR OCCURED\n");
        exit(0);
    }
}

__global__ void compute_kernel(
    double* u_new, double* u,
    Grid grid,
    mydim3<int> bsize,
    mydim3<double> h)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

    double inv_hxsqr = 1.0 / (h.x * h.x);
    double inv_hysqr = 1.0 / (h.y * h.y);
    double inv_hzsqr = 1.0 / (h.z * h.z);

    for (int i = id_x; i < bsize.x; i += offset_x)
        for (int j = id_y; j < bsize.y; j += offset_y)
            for (int k = id_z; k < bsize.z; k += offset_z) {

                double num = (u[grid.cell_absolute_id(i + 1, j, k)] + u[grid.cell_absolute_id(i - 1, j, k)]) * inv_hxsqr
                    + (u[grid.cell_absolute_id(i, j + 1, k)] + u[grid.cell_absolute_id(i, j - 1, k)]) * inv_hysqr
                    + (u[grid.cell_absolute_id(i, j, k + 1)] + u[grid.cell_absolute_id(i, j, k - 1)]) * inv_hzsqr;
                double denum = 2.0 * (inv_hxsqr + inv_hysqr + inv_hzsqr);

                u_new[grid.cell_absolute_id(i, j, k)] = num / denum;
            }
}

__global__ void abs_error_kernel(double* u1, double* u2, Grid grid, mydim3<int> bsize)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

    for (int i = id_x - 1; i < bsize.x + 1; i += offset_x) {
        for (int j = id_y - 1; j < bsize.y + 1; j += offset_y) {
            for (int k = id_z - 1; k < bsize.z + 1; k += offset_z) {
                u1[grid.cell_absolute_id(i, j, k)] = fabsf(u1[grid.cell_absolute_id(i, j, k)] - u2[grid.cell_absolute_id(i, j, k)]);
            }
        }
    }
}