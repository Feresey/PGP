#include "kernels.hpp"

DeviceProblem::DeviceProblem(const Grid& grid, const int kernel_grid_dim = 8, const int kernel_block_dim = 8)
    : grid(grid)
    , kernel_grid_dim(kernel_grid_dim)
    , kernel_block_dim(kernel_block_dim)
{
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

// void get_border_multigpu(double* dst, double* dev_dst, double** src, double** buffers, double** dev_buffers,
//     int border, std::vector<mydim3<int>> sizes, int device_count, int split_type)
// {
//     dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
//     dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

//     if (split_type == TOP_BOTTOM) {
//         if (border == TOP || border == BOTTOM) {
//             int src_index = border == BOTTOM ? 0 : device_count - 1;
//             int border_index = border == BOTTOM ? 0 : block_sizes_z[device_count - 1] - 1;
//             CUDA_ERR(cudaSetDevice(src_index));
//             START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
//                 block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index, UP_DOWN)));
//             CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyDeviceToHost));
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, LEFT_RIGHT)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
//             }
//             vertical_stack(dst, buffers, block_sizes_y[0], block_sizes_z, device_count);
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, FRONT_BACK)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
//             }
//             vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_z, device_count);
//         }
//     } else if (split_type == LEFT_RIGHT) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? 0 : block_sizes_z[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, UP_DOWN)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
//             }
//             horizontal_stack(dst, buffers, block_sizes_x, block_sizes_y[0], device_count);
//         } else if (border == LEFT || border == RIGHT) {
//             int src_index = border == LEFT ? 0 : device_count - 1;
//             int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
//             CUDA_ERR(cudaSetDevice(src_index));
//             START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
//                 block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index, LEFT_RIGHT)));
//             CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, FRONT_BACK)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
//             }
//             horizontal_stack(dst, buffers, block_sizes_x, block_sizes_z[0], device_count);
//         }
//     } else if (split_type == FRONT_BACK) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? 0 : block_sizes_z[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, UP_DOWN)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
//             }
//             vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_y, device_count);
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? 0 : block_sizes_x[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, LEFT_RIGHT)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
//             }
//             horizontal_stack(dst, buffers, block_sizes_y, block_sizes_z[0], device_count);
//         } else if (border == FRONT || border == BACK) {
//             int src_index = border == FRONT ? 0 : device_count - 1;
//             int border_index = border == FRONT ? 0 : block_sizes_y[device_count - 1] - 1;
//             CUDA_ERR(cudaSetDevice(src_index));
//             START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
//                 block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index, FRONT_BACK)));
//             CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
//         }
//     } else {
//         printf("ERROR OCCURED\n");
//         exit(0);
//     }
// }

// void set_border_multigpu(double** dst, double* src, double* dev_src, double** buffers, double** dev_buffers,
//     int border, int* block_sizes_x, int* block_sizes_y, int* block_sizes_z, int device_count, int split_type)
// {
//     dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
//     dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

//     if (split_type == UP_DOWN) {
//         if (border == TOP || border == BOTTOM) {
//             int src_index = border == BOTTOM ? 0 : device_count - 1;
//             int border_index = border == BOTTOM ? -1 : block_sizes_z[device_count - 1];
//             CUDA_ERR(cudaSetDevice(src_index));
//             CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyHostToDevice));
//             START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
//                 block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index, UP_DOWN)));
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
//             vertical_unstack(src, buffers, block_sizes_y[0], block_sizes_z, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, LEFT_RIGHT)));
//             }
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
//             vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_z, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, FRONT_BACK)));
//             }
//         }
//     } else if (split_type == LEFT_RIGHT) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? -1 : block_sizes_z[device_count - 1];
//             horizontal_unstack(src, buffers, block_sizes_x, block_sizes_y[0], device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, UP_DOWN)));
//             }
//         } else if (border == LEFT || border == RIGHT) {
//             int src_index = border == LEFT ? 0 : device_count - 1;
//             int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
//             CUDA_ERR(cudaSetDevice(src_index));
//             CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
//             START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
//                 block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index, LEFT_RIGHT)));
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
//             horizontal_unstack(src, buffers, block_sizes_x, block_sizes_z[0], device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, FRONT_BACK)));
//             }
//         }
//     } else if (split_type == FRONT_BACK) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? -1 : block_sizes_z[device_count - 1];
//             vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_y, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, UP_DOWN)));
//             }
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? -1 : block_sizes_x[device_count - 1];
//             horizontal_unstack(src, buffers, block_sizes_y, block_sizes_z[0], device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index, LEFT_RIGHT)));
//             }
//         } else if (border == FRONT || border == BACK) {
//             int src_index = border == FRONT ? 0 : device_count - 1;
//             int border_index = border == FRONT ? -1 : block_sizes_y[device_count - 1];
//             CUDA_ERR(cudaSetDevice(src_index));
//             CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
//             START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
//                 block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index, FRONT_BACK)));
//         }
//     } else {
//         printf("ERROR OCCURED\n");
//         exit(0);
//     }
// }

// void get_intergpu_border(double* dst, double* src, double* dev_buffer, int index, mydim3<int> bsize, border_tag split_type)
// {
//     dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
//     dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

//     START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, bsize.x, bsize.y, bsize.z, index, split_type)));
//     CUDA_ERR(cudaMemcpy(dst, dev_buffer, sizeof(double) * bsize.x * bsize.y, cudaMemcpyDeviceToHost));
// }

// void set_intergpu_border(double* dst, double* src, double* dev_buffer, int index, mydim3<int> bsize, border_tag split_type)
// {
//     dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
//     dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

//     CUDA_ERR(cudaMemcpy(dev_buffer, src, sizeof(double) * bsize.x * bsize.y, cudaMemcpyHostToDevice));
//     START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, bsize.x, bsize.y, bsize.z, index, split_type)));
// }
