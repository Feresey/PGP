#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.cuh"

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

#define THREAD_INDEX(idx) ((WARP_SIZE + 1) * ((idx) / WARP_SIZE) + ((idx) % WARP_SIZE))

#define SWAP_IF(arr, x, y) \
    if (arr[x] > arr[y])   \
    SWAP(arr, x, y)

__global__ void int_memset(int* dev_arr, int n, int val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += offset)
        dev_arr[i] = val;
}

__global__ void sort_blocks(int* dev_arr)
{
    __shared__ int shared[BLOCK_SIZE + (BLOCK_SIZE / WARP_SIZE) + 1];

    const unsigned int thread_id = threadIdx.x,
                       block_offset = blockIdx.x * BLOCK_SIZE,
                       second_half_idx = thread_id + BLOCK_SIZE / 2;

    shared[THREAD_INDEX(thread_id)] = dev_arr[thread_id + block_offset];
    shared[THREAD_INDEX(second_half_idx)] = dev_arr[second_half_idx + block_offset];

    if (thread_id == 0) {
        shared[THREAD_INDEX(BLOCK_SIZE)] = INT_MAX;
    }

    __syncthreads();

    int swap1_idx1 = 2 * thread_id,
        swap1_idx2 = 2 * thread_id + 1,
        swap2_idx2 = 2 * thread_id + 2;

    swap1_idx1 = THREAD_INDEX(swap1_idx1);
    swap1_idx2 = THREAD_INDEX(swap1_idx2);
    swap2_idx2 = THREAD_INDEX(swap2_idx2);

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        SWAP_IF(shared, swap1_idx1, swap1_idx2);
        __syncthreads();
        SWAP_IF(shared, swap1_idx2, swap2_idx2);
        __syncthreads();
    }

    __syncthreads();

    dev_arr[thread_id + block_offset] = shared[THREAD_INDEX(thread_id)];
    dev_arr[thread_id + block_offset + BLOCK_SIZE / 2] = shared[THREAD_INDEX(second_half_idx)];
}

__global__ void merge(int* dev_arr, int iter, int type)
{
    __shared__ int shared[BLOCK_SIZE + (BLOCK_SIZE / WARP_SIZE)];

    const unsigned int thread_id = threadIdx.x,
                       block_offset = blockIdx.x * BLOCK_SIZE,
                       load_second_half_idx = BLOCK_SIZE - thread_id - 1;

    shared[THREAD_INDEX(thread_id)] = dev_arr[thread_id + block_offset];
    shared[THREAD_INDEX(load_second_half_idx)] = dev_arr[thread_id + block_offset + BLOCK_SIZE / 2];

    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        int i = thread_id / stride,
            j = thread_id % stride;

        int swap_idx1 = 2 * stride * i + j,
            swap_idx2 = 2 * stride * i + j + stride;

        swap_idx1 = THREAD_INDEX(swap_idx1);
        swap_idx2 = THREAD_INDEX(swap_idx2);

        __syncthreads();

        SWAP_IF(shared, swap_idx1, swap_idx2);
    }

    __syncthreads();
    
    dev_arr[thread_id + block_offset] = shared[THREAD_INDEX(thread_id)];
    int store_second_half_idx = thread_id + BLOCK_SIZE / 2;
    dev_arr[thread_id + block_offset + BLOCK_SIZE / 2] = shared[THREAD_INDEX(store_second_half_idx)];
}

int nearest_size(int num, int prod)
{
    int mod = num % prod;
    return num + (prod - ((mod == 0) ? prod : mod));
}

void block_odd_even_sort(int* arr, int n)
{
    // размер, кратный размеру блока
    int dev_n = nearest_size(n, BLOCK_SIZE);
    int n_blocks = dev_n / BLOCK_SIZE;

    int* dev_arr;
    CSC(cudaMalloc(&dev_arr, dev_n * sizeof(int)));
    CSC(cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice));

    // гарантия что элементы, которые получились из за расширения окажутся в начале отсортированного массива
    START_KERNEL((int_memset<<<1, BLOCK_SIZE>>>(dev_arr + n, (dev_n - n), INT_MIN)));
    // подготовка блоков
    START_KERNEL((sort_blocks<<<n_blocks, BLOCK_SIZE / 2>>>(dev_arr)));

    if (n_blocks == 1) {
        CSC(cudaMemcpy(arr, dev_arr + (dev_n - n), n * sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaFree(dev_arr));
        return;
    }

    for (int iter = 0; iter < n_blocks; ++iter) {
        START_KERNEL((merge<<<n_blocks - 1, BLOCK_SIZE / 2>>>(dev_arr + BLOCK_SIZE / 2, iter, 0)));
        START_KERNEL((merge<<<n_blocks, BLOCK_SIZE / 2>>>(dev_arr, iter, 1)));
    }

    CSC(cudaMemcpy(arr, dev_arr + (dev_n - n), n * sizeof(int), cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_arr));
}

int main()
{
    uint32_t size = scan_4();

    int* arr = (int*)malloc(size * sizeof(int));
    for (uint32_t i = 0; i < size; ++i) {
        arr[i] = int(scan_4());
    }

    block_odd_even_sort(arr, size);

    print_arr(stdout, arr, size);

    free(arr);
    return 0;
}