#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.cuh"

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

// попытка выровнять блоки
#define THREAD_INDEX(idx) ((WARP_SIZE + 1) * ((idx) / WARP_SIZE) + ((idx) % WARP_SIZE))

#define SWAP_IF(arr, x, y) \
    if (arr[x] > arr[y])   \
    SWAP(arr, x, y)

__global__ void sort_blocks_even_odd(int* dev_arr)
{
    // хвостик от выравнивания банков
    __shared__ int shared[THREAD_INDEX(BLOCK_SIZE + 1)];

    const unsigned int thread_id = threadIdx.x,
                       block_offset = blockIdx.x * BLOCK_SIZE,
                       second_half_idx = thread_id + BLOCK_SIZE / 2;

    shared[THREAD_INDEX(thread_id)] = dev_arr[thread_id + block_offset];
    shared[THREAD_INDEX(second_half_idx)] = dev_arr[second_half_idx + block_offset];

    if (thread_id == 0) {
        shared[THREAD_INDEX(BLOCK_SIZE)] = INT_MAX;
    }

    __syncthreads();

    int swap1 = 2 * thread_id,
        swap2 = 2 * thread_id + 1,
        swap3 = 2 * thread_id + 2;

    swap1 = THREAD_INDEX(swap1);
    swap2 = THREAD_INDEX(swap2);
    swap3 = THREAD_INDEX(swap3);

    // да, я знаю что тут конфликт
    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        __syncthreads();
        SWAP_IF(shared, swap1, swap2);
        __syncthreads();
        SWAP_IF(shared, swap2, swap3);
    }
    __syncthreads();

    dev_arr[thread_id + block_offset] = shared[THREAD_INDEX(thread_id)];
    dev_arr[thread_id + block_offset + BLOCK_SIZE / 2] = shared[THREAD_INDEX(second_half_idx)];
}

__global__ void bitonic_merge(int* dev_arr)
{
    __shared__ int shared[BLOCK_SIZE];

    const unsigned int thread_id = threadIdx.x,
                       block_offset = blockIdx.x * BLOCK_SIZE,
                       load_second_half_idx = BLOCK_SIZE - thread_id - 1,
                       store_second_half_idx = thread_id + BLOCK_SIZE / 2;

    shared[thread_id] = dev_arr[thread_id + block_offset];
    shared[load_second_half_idx] = dev_arr[thread_id + block_offset + BLOCK_SIZE / 2];

    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        int i = thread_id / stride,
            j = thread_id % stride;

        int swap1 = 2 * stride * i + j,
            swap2 = 2 * stride * i + j + stride;

        __syncthreads();

        SWAP_IF(shared, swap1, swap2);
    }

    __syncthreads();

    dev_arr[thread_id + block_offset] = shared[thread_id];
    dev_arr[thread_id + block_offset + BLOCK_SIZE / 2] = shared[store_second_half_idx];
}

__global__ void dummy_memset(int* dev_arr, const uint32_t n, const int val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += offset) {
        dev_arr[i] = val;
    }
}

uint32_t nearest_size(const uint32_t num, const uint32_t prod)
{
    uint32_t mod = num % prod;
    return num + (prod - ((mod == 0) ? prod : mod));
}

void sort(int* arr, const uint32_t n)
{
    // размер, кратный размеру блока
    const uint32_t dev_n = nearest_size(n, BLOCK_SIZE);
    const uint32_t n_blocks = dev_n / BLOCK_SIZE;

    int* dev_arr;
    CSC(cudaMalloc(&dev_arr, dev_n * sizeof(int)));
    CSC(cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice));

    // гарантия что элементы, которые получились из за расширения окажутся в начале отсортированного массива
    START_KERNEL((dummy_memset<<<1, BLOCK_SIZE>>>(dev_arr + n, (dev_n - n), INT_MIN)));

    // предварительная сортировка блоков
    START_KERNEL((sort_blocks_even_odd<<<n_blocks, BLOCK_SIZE / 2>>>(dev_arr)));

    // ну если массив влез в 1 блок то зачем его сортировать ещё раз?
    if (n_blocks == 1) {
        goto END;
    }

    // Тут есть n_blocks отсортированных неубывающих последовательностей.
    // Чтобы все элементы встали на свои места, будут сортироваться половинки блоков:
    // Правая половина первого и левая половина второго; правая половина второго и левая третьего; ...
    // Чудесное совпадение, что битоническое слияние как раз на такое и рассчитано.
    for (uint32_t iter = 0; iter < n_blocks; ++iter) {
        // BLOCK_SIZE/2 потому что каждый поток владеет
        // одним элементом от первого отсортированного блока и одним от второго
        START_KERNEL((bitonic_merge<<<n_blocks - 1, BLOCK_SIZE / 2>>>(dev_arr + BLOCK_SIZE / 2)));
        START_KERNEL((bitonic_merge<<<n_blocks, BLOCK_SIZE / 2>>>(dev_arr)));
    }

END:
    // обрезание хвостика, который добавлялся для выравнивания. Можно было конечно вписать INT_MAX, но так не интересно.
    CSC(cudaMemcpy(arr, dev_arr + (dev_n - n), n * sizeof(int), cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_arr));
}

#include <time.h>

// call this function to start a nanosecond-resolution timer
struct timespec timer_start()
{
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

// call this function to end a timer, returning nanoseconds elapsed as a long
long timer_end(struct timespec start_time)
{
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}

int main()
{
    // как же неудобно работать с little/big endian в СИ
    uint32_t size = scan_4();

    int* arr = (int*)malloc(size * sizeof(int));
    for (uint32_t i = 0; i < size; ++i) {
        arr[i] = int(scan_4());
    }

    struct timespec start_time = timer_start();
    sort(arr, size);
    fprintf(stderr, "sort time: %d\n", timer_end(start_time));
    print_arr(stdout, arr, size);

    free(arr);
    return 0;
}