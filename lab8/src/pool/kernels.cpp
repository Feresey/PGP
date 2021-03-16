#include "kernels.hpp"

dim3_type layer_tag_to_dim3_type(layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return DIM3_TYPE_X;
    case FRONT_BACK:
        return DIM3_TYPE_Y;
    case VERTICAL:
        return DIM3_TYPE_Z;
    }
}

layer_tag dim3_type_to_layer_tag(dim3_type type)
{
    switch (type) {
    default:
    case DIM3_TYPE_X:
        return LEFT_RIGHT;
    case DIM3_TYPE_Y:
        return FRONT_BACK;
    case DIM3_TYPE_Z:
        return VERTICAL;
    }
}

DeviceProblem::DeviceProblem(
    const BlockGrid& grid, const int n_devices,
    const int kernel_grid_dim, const int kernel_block_dim)
    : grid(grid)
    , n_devices(n_devices)
    , kernel_grid_dim(kernel_grid_dim)
    , kernel_block_dim(kernel_block_dim)
{
}

// void horizontal_stack(double* dst, double** buffers, int* horizontal_sizes, int vertical_size, int device_count)
// {
//     int total_horizontal_size = 0;
//     for (int d = 0; d < device_count; ++d)
//         total_horizontal_size += horizontal_sizes[d];

//     for (int d = 0; d < device_count; ++d) {
//         for (int j = 0; j < vertical_size; ++j) {
//             for (int i = 0; i < horizontal_sizes[d]; ++i) {
//                 dst[total_horizontal_size * j + d * horizontal_sizes[0] + i] = buffers[d][j * horizontal_sizes[d] + i];
//             }
//         }
//     }
// }

// void vertical_stack(double* dst, double** buffers, int horizontal_size, int* vertical_sizes, int device_count)
// {
//     int offset = 0;
//     for (int d = 0; d < device_count; ++d) {
//         for (int j = 0; j < vertical_sizes[d]; ++j) {
//             for (int i = 0; i < horizontal_size; ++i) {
//                 dst[offset + j * horizontal_size + i] = buffers[d][j * horizontal_size + i];
//             }
//         }
//         offset += vertical_sizes[d] * horizontal_size;
//     }
// }

// void horizontal_unstack(double* src, double** buffers, int* horizontal_sizes, int vertical_size, int device_count)
// {
//     int total_horizontal_size = 0;
//     for (int d = 0; d < device_count; ++d)
//         total_horizontal_size += horizontal_sizes[d];

//     for (int d = 0; d < device_count; ++d) {
//         for (int j = 0; j < vertical_size; ++j) {
//             for (int i = 0; i < horizontal_sizes[d]; ++i) {
//                 buffers[d][j * horizontal_sizes[d] + i] = src[total_horizontal_size * j + d * horizontal_sizes[0] + i];
//             }
//         }
//     }
// }

// void vertical_unstack(double* src, double** buffers, int horizontal_size, int* vertical_sizes, int device_count)
// {
//     int offset = 0;
//     for (int d = 0; d < device_count; ++d) {
//         for (int j = 0; j < vertical_sizes[d]; ++j) {
//             for (int i = 0; i < horizontal_size; ++i) {
//                 buffers[d][j * horizontal_size + i] = src[offset + j * horizontal_size + i];
//             }
//         }
//         offset += vertical_sizes[d] * horizontal_size;
//     }
// }

// void get_border_multigpu(
//     double* dst, double* dev_dst,
//     double** src, double** buffers, double** dev_buffers,
//     side_tag border, mydim3<std::vector<int>> sizes, int device_count, layer_tag split_type)
// {
//     auto get_is_lower = [border](layer_tag split_type) -> bool {
//         switch (split_type) {
//         case VERTICAL:
//             return (border == BOTTOM);
//         case LEFT_RIGHT:
//             return (border == LEFT);
//         case FRONT_BACK:
//             return (border == FRONT);
//         default:
//             return -1;
//         }
//     };

//     // особый случай, когда нужна граница, принадлежащая только одной GPU
//     if ((border & split_type) != 0) {
//         bool is_lower = get_is_lower(split_type);

//         const int one_shot_idx = is_lower ? 0 : (device_count - 1);
//         const int border_idx = is_lower ? 0 : (sizes.z[device_count - 1] - 1);

//         // TODO get border

//         return;
//     }

//     switch (split_type) {
//     case VERTICAL:
//         bool is_lower = get_is_lower((border & LEFT_RIGHT) ? LEFT_RIGHT : FRONT_BACK);
//         break;
//     case LEFT_RIGHT:
//         bool is_lower = get_is_lower((border & FRONT_BACK) ? FRONT_BACK : VERTICAL);
//         break;
//     case FRONT_BACK:
//         bool is_lower = get_is_lower((border & VERTICAL) ? VERTICAL : LEFT_RIGHT);
//         break;
//     }

//     if (split_type == VERTICAL) {
//         if (border == TOP || border == BOTTOM) {
//             int src_index = border == BOTTOM ? 0 : device_count - 1;
//             int border_index = border == BOTTOM ? 0 : sizes.z[device_count - 1] - 1;

//             CUDA_ERR(cudaSetDevice(src_index));
//             START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
//                 sizes.x[src_index], sizes.y[src_index], sizes.z[src_index], border_index, UP_DOWN)));
//             CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * sizes.x[0] * sizes.y[0], cudaMemcpyDeviceToHost));
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? 0 : sizes.x[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     sizes.x[d], sizes.y[d], sizes.z[d], border_index, LEFT_RIGHT)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * sizes.y[d] * sizes.z[d], cudaMemcpyDeviceToHost));
//             }
//             vertical_stack(dst, buffers, sizes.y[0], sizes.z, device_count);
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? 0 : sizes.y[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     sizes.x[d], sizes.y[d], sizes.z[d], border_index, FRONT_BACK)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * sizes.x[d] * sizes.z[d], cudaMemcpyDeviceToHost));
//             }
//             vertical_stack(dst, buffers, sizes.x[0], sizes.z, device_count);
//         }
//     } else if (split_type == LEFT_RIGHT) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? 0 : sizes.z[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     sizes.x[d], sizes.y[d], sizes.z[d], border_index, UP_DOWN)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * sizes.x[d] * sizes.y[d], cudaMemcpyDeviceToHost));
//             }
//             horizontal_stack(dst, buffers, sizes, sizes.x[0], device_count);
//         } else if (border == LEFT || border == RIGHT) {
//             int src_index = border == LEFT ? 0 : device_count - 1;
//             int border_index = border == LEFT ? 0 : sizes.y[device_count - 1] - 1;
//             CUDA_ERR(cudaSetDevice(src_index));
//             START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
//                 sizes.x[src_index], sizes.x[src_index], sizes.y[src_index], border_index, LEFT_RIGHT)));
//             CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * sizes.z[0] * sizes.y[0], cudaMemcpyDeviceToHost));
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? 0 : sizes.z[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     sizes.y[d], sizes.x[d], sizes.y[d], border_index, FRONT_BACK)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * sizes.z[d] * sizes.x[d], cudaMemcpyDeviceToHost));
//             }
//             horizontal_stack(dst, buffers, sizes, sizes.z[0], device_count);
//         }
//     } else if (split_type == FRONT_BACK) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? 0 : sizes.x[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     sizes.z[d], sizes.z[d], sizes.x[d], border_index, UP_DOWN)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * sizes.y[d] * sizes.z[d], cudaMemcpyDeviceToHost));
//             }
//             vertical_stack(dst, buffers, sizes.x[0], sizes.y, device_count);
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? 0 : sizes.x[device_count - 1] - 1;
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
//                     sizes.y[d], sizes.x[d], sizes.x[d], border_index, LEFT_RIGHT)));
//                 CUDA_ERR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * sizes.y[d] * sizes.z[d], cudaMemcpyDeviceToHost));
//             }
//             horizontal_stack(dst, buffers, sizes, sizes.y[0], device_count);
//         } else if (border == FRONT || border == BACK) {
//             int src_index = border == FRONT ? 0 : device_count - 1;
//             int border_index = border == FRONT ? 0 : sizes.z[device_count - 1] - 1;
//             CUDA_ERR(cudaSetDevice(src_index));
//             START_KERNEL((get_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
//                 sizes.y[src_index], block_sizesy[src_index].z, block_sizesz[src_index].y, border_index, FRONT_BACK)));
//             CUDA_ERR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizesx[0].x * block_sizesz[0]._, cudaMemcpyDeviceToHost));
//         }
//     } else {
//         printf("ERROR OCCURED\n");
//         exit(0);
//     }
// }

// void set_border_multigpu(double** dst, double* src, double* dev_src, double** buffers, double** dev_buffers,
//     int border, int* block_sizesx._, int* block_sizesy._, int* block_sizesz._, int device_count, int split_type)
// {
//     dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
//     dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

//     if (split_type == UP_DOWN) {
//         if (border == TOP || border == BOTTOM) {
//             int src_index = border == BOTTOM ? 0 : device_count - 1;
//             int border_index = border == BOTTOM ? -1 : block_sizesz[device_count - 1]._;
//             CUDA_ERR(cudaSetDevice(src_index));
//             CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizesx[0]._ * block_sizesy[0]._, cudaMemcpyHostToDevice));
//             START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
//                 block_sizesx[src_index]._, block_sizesy[src_index]._, block_sizesz[src_index]._, border_index, UP_DOWN)));
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? -1 : block_sizesx[device_count - 1]._;
//             vertical_unstack(src, buffers, block_sizesy[0]._, block_sizesz._, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizesy[d]._ * block_sizesz[d]._, cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizesx[d]._, block_sizesy[d]._, block_sizesz[d]._, border_index, LEFT_RIGHT)));
//             }
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? -1 : block_sizesy[device_count - 1]._;
//             vertical_unstack(src, buffers, block_sizesx[0]._, block_sizesz._, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizesx[d]._ * block_sizesz[d]._, cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizesx[d]._, block_sizesy[d]._, block_sizesz[d]._, border_index, FRONT_BACK)));
//             }
//         }
//     } else if (split_type == LEFT_RIGHT) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? -1 : block_sizesz[device_count - 1]._;
//             horizontal_unstack(src, buffers, block_sizesx, block_sizesy[0]._, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizesx[d]._ * block_sizesy[d]._, cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizesx[d]._, block_sizesy[d]._, block_sizesz[d]._, border_index, UP_DOWN)));
//             }
//         } else if (border == LEFT || border == RIGHT) {
//             int src_index = border == LEFT ? 0 : device_count - 1;
//             int border_index = border == LEFT ? -1 : block_sizesx[device_count - 1]._;
//             CUDA_ERR(cudaSetDevice(src_index));
//             CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizesy[0]._ * block_sizesz[0]._, cudaMemcpyHostToDevice));
//             START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
//                 block_sizesx[src_index]._, block_sizesy[src_index]._, block_sizesz[src_index]._, border_index, LEFT_RIGHT)));
//         } else if (border == FRONT || border == BACK) {
//             int border_index = border == FRONT ? -1 : block_sizesy[device_count - 1]._;
//             horizontal_unstack(src, buffers, block_sizesx, block_sizesz[0]._, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizesx[d]._ * block_sizesz[d]._, cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizesx[d]._, block_sizesy[d]._, block_sizesz[d]._, border_index, FRONT_BACK)));
//             }
//         }
//     } else if (split_type == FRONT_BACK) {
//         if (border == TOP || border == BOTTOM) {
//             int border_index = border == BOTTOM ? -1 : block_sizesz[device_count - 1]._;
//             vertical_unstack(src, buffers, block_sizesx[0]._, block_sizesy._, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizesx[d]._ * block_sizesy[d]._, cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizesx[d]._, block_sizesy[d]._, block_sizesz[d]._, border_index, UP_DOWN)));
//             }
//         } else if (border == LEFT || border == RIGHT) {
//             int border_index = border == LEFT ? -1 : block_sizesx[device_count - 1]._;
//             horizontal_unstack(src, buffers, block_sizesy, block_sizesz[0]._, device_count);
//             for (int d = 0; d < device_count; ++d) {
//                 CUDA_ERR(cudaSetDevice(d));
//                 CUDA_ERR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizesy[d]._ * block_sizesz[d]._, cudaMemcpyHostToDevice));
//                 START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
//                     block_sizesx[d]._, block_sizesy[d]._, block_sizesz[d]._, border_index, LEFT_RIGHT)));
//             }
//         } else if (border == FRONT || border == BACK) {
//             int src_index = border == FRONT ? 0 : device_count - 1;
//             int border_index = border == FRONT ? -1 : block_sizesy[device_count - 1]._;
//             CUDA_ERR(cudaSetDevice(src_index));
//             CUDA_ERR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizesx[0]._ * block_sizesz[0]._, cudaMemcpyHostToDevice));
//             START_KERNEL((set_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
//                 block_sizesx[src_index]._, block_sizesy[src_index]._, block_sizesz[src_index]._, border_index, FRONT_BACK)));
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
