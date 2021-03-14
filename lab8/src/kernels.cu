#include "dim3/dim3.hpp"
#include "helpers.cuh"

#define LEFT 0
#define RIGHT 1
#define BACK 2
#define FRONT 3
#define DOWN 4
#define UP 5

#define BORDER_OPERATIONS_GRID_DIM 32
#define BORDER_OPERATIONS_BLOCK_DIM 32

#define GRID_DIM 8

#define MULTIPLE_GPU_CRITERIA 100000

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
                temp = buf[grid.cell_idx(border_idx, a ,b);
                break;
            case FRONT_BACK:
                temp = buf[grid.cell_idx(a, border_idx, b);
                break;
            case UP_DOWN:
                temp = buf[grid.cell_idx(a, b, border_idx);
                break;
            }

            out[j * block_size_x + i] = temp;
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
                dest_idx = grid.cell_idx(border_idx, a, b);
                break;
            case FRONT_BACK:
                dest_idx = grid.cell_idx(a, border_idx, , b);
                break;
            case UP_DOWN:
                dest_idx = grid.cell_idx(a, b, border_idx);
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

    if (split_type == UPDOWN) {
        if (border == UP || border == DOWN) {
            int src_index = border == DOWN ? 0 : device_count - 1;
            int border_index = border == DOWN ? 0 : block_sizes_z[device_count - 1] - 1;
            CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
            get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyDeviceToHost));
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            vertical_stack(dst, buffers, block_sizes_y[0], block_sizes_z, device_count);
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_z, device_count);
        }
    } else if (split_type == LEFTRIGHT) {
        if (border == UP || border == DOWN) {
            int border_index = border == DOWN ? 0 : block_sizes_z[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
            }
            horizontal_stack(dst, buffers, block_sizes_x, block_sizes_y[0], device_count);
        } else if (border == LEFT || border == RIGHT) {
            int src_index = border == LEFT ? 0 : device_count - 1;
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
            CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
            get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            horizontal_stack(dst, buffers, block_sizes_x, block_sizes_z[0], device_count);
        }
    } else if (split_type == FRONTBACK) {
        if (border == UP || border == DOWN) {
            int border_index = border == DOWN ? 0 : block_sizes_z[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
            }
            vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_y, device_count);
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
            }
            horizontal_stack(dst, buffers, block_sizes_y, block_sizes_z[0], device_count);
        } else if (border == FRONT || border == BACK) {
            int src_index = border == FRONT ? 0 : device_count - 1;
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
            CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
            get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
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

    if (split_type == UPDOWN) {
        if (border == UP || border == DOWN) {
            int src_index = border == DOWN ? 0 : device_count - 1;
            int border_index = border == DOWN ? -1 : block_sizes_z[device_count - 1];
            CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyHostToDevice));
            set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
            vertical_unstack(src, buffers, block_sizes_y[0], block_sizes_z, device_count);
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
            }
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
            vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_z, device_count);
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
            }
        }
    } else if (split_type == LEFTRIGHT) {
        if (border == UP || border == DOWN) {
            int border_index = border == DOWN ? -1 : block_sizes_z[device_count - 1];
            horizontal_unstack(src, buffers, block_sizes_x, block_sizes_y[0], device_count);
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
                set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
            }
        } else if (border == LEFT || border == RIGHT) {
            int src_index = border == LEFT ? 0 : device_count - 1;
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
            CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
            set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
        } else if (border == FRONT || border == BACK) {
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
            horizontal_unstack(src, buffers, block_sizes_x, block_sizes_z[0], device_count);
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
            }
        }
    } else if (split_type == FRONTBACK) {
        if (border == UP || border == DOWN) {
            int border_index = border == DOWN ? -1 : block_sizes_z[device_count - 1];
            vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_y, device_count);
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
                set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
            }
        } else if (border == LEFT || border == RIGHT) {
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
            horizontal_unstack(src, buffers, block_sizes_y, block_sizes_z[0], device_count);
            for (int d = 0; d < device_count; ++d) {
                CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
                CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
                set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
                    block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
            }
        } else if (border == FRONT || border == BACK) {
            int src_index = border == FRONT ? 0 : device_count - 1;
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
            CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
            set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
                block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
        }
    } else {
        printf("ERROR OCCURED\n");
        exit(0);
    }
}

int get_last_index(int split_type, int* block_sizes_x, int* block_sizes_y, int* block_sizes_z, int device)
{
    int last_index;
    if (split_type == UPDOWN)
        last_index = block_sizes_z[device] - 1;
    if (split_type == LEFTRIGHT)
        last_index = block_sizes_x[device] - 1;
    if (split_type == FRONTBACK)
        last_index = block_sizes_y[device] - 1;
    return last_index;
}

void get_intergpu_border(double* dst, double* src, double* dev_buffer, int index, int block_size_x, int block_size_y, int block_size_z, int split_type)
{
    dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
    dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

    if (split_type == UPDOWN) {
        get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, block_size_x, block_size_y, block_size_z, index);
        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_buffer, sizeof(double) * block_size_x * block_size_y, cudaMemcpyDeviceToHost));
        CHECK_CUDA_KERNEL_ERROR();
    } else if (split_type == LEFTRIGHT) {
        get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, block_size_x, block_size_y, block_size_z, index);
        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_buffer, sizeof(double) * block_size_y * block_size_z, cudaMemcpyDeviceToHost));
        CHECK_CUDA_KERNEL_ERROR();
    } else if (split_type == FRONTBACK) {
        get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, block_size_x, block_size_y, block_size_z, index);
        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_buffer, sizeof(double) * block_size_x * block_size_z, cudaMemcpyDeviceToHost));
        CHECK_CUDA_KERNEL_ERROR();
    } else {
        printf("ERROR OCCURED\n");
        exit(0);
    }
}

void set_intergpu_border(double* dst, double* src, double* dev_buffer, int index, int block_size_x, int block_size_y, int block_size_z, int split_type)
{
    dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
    dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

    if (split_type == UPDOWN) {
        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffer, src, sizeof(double) * block_size_x * block_size_y, cudaMemcpyHostToDevice));
        set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, block_size_x, block_size_y, block_size_z, index);
        CHECK_CUDA_KERNEL_ERROR();
    } else if (split_type == LEFTRIGHT) {
        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffer, src, sizeof(double) * block_size_y * block_size_z, cudaMemcpyHostToDevice));
        set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, block_size_x, block_size_y, block_size_z, index);
        CHECK_CUDA_KERNEL_ERROR();
    } else if (split_type == FRONTBACK) {
        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffer, src, sizeof(double) * block_size_x * block_size_z, cudaMemcpyHostToDevice));
        set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, block_size_x, block_size_y, block_size_z, index);
        CHECK_CUDA_KERNEL_ERROR();
    } else {
        printf("ERROR OCCURED\n");
        exit(0);
    }
}

__global__ void compute_kernel(double* u_new, double* u, int block_size_x, int block_size_y, int block_size_z, double hx, double hy, double hz)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

    for (int i = id_x; i < block_size_x; i += offset_x)
        for (int j = id_y; j < block_size_y; j += offset_y)
            for (int k = id_z; k < block_size_z; k += offset_z) {
                double inv_hxsqr = 1.0 / (hx * hx);
                double inv_hysqr = 1.0 / (hy * hy);
                double inv_hzsqr = 1.0 / (hz * hz);

                double num = (u[cell_index(i + 1, j, k)] + u[cell_index(i - 1, j, k)]) * inv_hxsqr
                    + (u[cell_index(i, j + 1, k)] + u[cell_index(i, j - 1, k)]) * inv_hysqr
                    + (u[cell_index(i, j, k + 1)] + u[cell_index(i, j, k - 1)]) * inv_hzsqr;
                double denum = 2.0 * (inv_hxsqr + inv_hysqr + inv_hzsqr);

                u_new[cell_index(i, j, k)] = num / denum;
            }
}

__global__ void abs_error_kernel(double* u1, double* u2, int block_size_x, int block_size_y, int block_size_z)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

    for (int i = id_x - 1; i < block_size_x + 1; i += offset_x)
        for (int j = id_y - 1; j < block_size_y + 1; j += offset_y)
            for (int k = id_z - 1; k < block_size_z + 1; k += offset_z) {
                u1[cell_index(i, j, k)] = fabsf(u1[cell_index(i, j, k)] - u2[cell_index(i, j, k)]);
            }
}