#include <algorithm>
#include <numeric>

#include "kernels.hpp"
#include "pool.hpp"

// split_by возвращает std::pair<part_size, rest>
split_by::split_by(int need_split, int n_parts, int min_part_size)
{
    part_size = need_split / n_parts;
    if (part_size < min_part_size) {
        int n_parts_normalized = need_split / min_part_size;
        // нет смысла делить на 1 целую часть и ещё 1 часть с "хвостиком"
        if (n_parts_normalized == 1) {
            this->part_size = need_split;
            this->rest = 0;
            this->n_parts = 1;
            return;
        }
        this->part_size = need_split / n_parts_normalized;
        this->rest = need_split % n_parts_normalized;
        this->n_parts = n_parts_normalized;
        return;
    }
    this->rest = need_split % n_parts;
    this->n_parts = n_parts;
}

// Нет смысла делить маленькие данные, проще (и быстрее) запихнуть их на один GPU
#define MIN_GPU_SPLIT_SIZE 5000

GPU_pool::GPU_pool(const Grid& grid, Task task)
    // данные для разных GPU будут разделяться по наибольшей стороне блока.
    // Да, можно сделать лучше.
    : split_type(dim3_type_to_layer_tag(grid.bsize.max_dim().get_type()))
    , grid(grid)
    , task(task)
{
    int n_devices = this->get_devices();

    auto max_elem = grid.bsize.max_dim();
    int max_dim = *max_elem;

    //@ сторона делится на n_devices частей, остаток добавляется последнему блоку@
    const split_by& split = split_by(max_dim, n_devices, MIN_GPU_SPLIT_SIZE);

    mydim3<int> init_dim;
    mydim3<int> rest_dim;
    for (auto elem = grid.bsize.begin(); elem != grid.bsize.end(); ++elem) {
        auto offset = std::distance(grid.bsize.begin(), elem);
        if (elem == max_elem) {
            init_dim[offset] = split.part_size;
            rest_dim[offset] = split.part_size + split.rest;
        } else {
            init_dim[offset] = grid.bsize[offset];
            rest_dim[offset] = grid.bsize[offset];
        }
    }

    int max_rest_dim = *rest_dim.max_dim();

    this->devices = std::vector<Elem>(size_t(split.n_parts), Elem(BlockGrid { init_dim }, max_rest_dim));
    this->devices.back() = Elem(BlockGrid { rest_dim }, max_rest_dim);

    this->init_devices(max_dim);
}

void GPU_pool::load_gpu_data(side_tag border)
{
    auto get_is_lower = [border](layer_tag split_type) -> bool {
        switch (split_type) {
        case VERTICAL:
            return (border == BOTTOM);
        case LEFT_RIGHT:
            return (border == LEFT);
        case FRONT_BACK:
            return (border == FRONT);
        default:
            return -1;
        }
    };

    // особый случай, когда нужна граница, принадлежащая только одной GPU
    if ((border & split_type) != 0) {
        bool is_lower = get_is_lower(split_type);

        const size_t one_shot_idx = is_lower ? 0UL : (devices.size() - 1UL);
        const int layer_idx = is_lower ? 0 : (devices.back().grid.bsize[side_tag_to_dim3_type(border)] - 1);

        // TODO get border

        auto& device = devices[one_shot_idx];
        device.load_border(layer_idx, split_type);

        return;
    }

    bool is_lower;

    switch (split_type) {
    case VERTICAL:
        is_lower = get_is_lower((border & LEFT_RIGHT) ? LEFT_RIGHT : FRONT_BACK);
        break;
    case LEFT_RIGHT:
        is_lower = get_is_lower((border & FRONT_BACK) ? FRONT_BACK : VERTICAL);
        break;
    case FRONT_BACK:
        is_lower = get_is_lower((border & VERTICAL) ? VERTICAL : LEFT_RIGHT);
        break;
    }

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
}

void GPU_pool::store_gpu_data(side_tag tag)
{
}

double GPU_pool::calc()
{
    double all_error = 0.0;
    for (size_t device_id = 0; device_id < devices.size();++device_id) {
        auto& device = devices[device_id];
        device.set_device(int(device_id));
        double local_err = device.compute(height);
        if (local_err > all_error) {
            all_error = local_err;
        }
    }
    return all_error;
}