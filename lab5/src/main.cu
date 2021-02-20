#include <byteswap.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.cuh"

#define THREADS 8
#define BLOCKS 8
#define NUM_VALS THREADS* BLOCKS

#define SWAP(a, b)      \
    do {                \
        int temp = (a); \
        (a) = (b);      \
        (b) = temp;     \
    } while (false)

uint32_t scan_4()
{
    uint32_t temp;
    scanf("%x", &temp);
    return __bswap_32(temp);
}

void print_arr(FILE* out, const int* arr, const uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        fprintf(out, "%08x", __bswap_32(arr[i]));
        if (i != size - 1) {
            fprintf(out, " ");
        }
    }
    fprintf(out, "\n");
}

#define BLOCK_SIZE 4
__global__ void kernel_bitonic_sort(int* arr)
{
    const uint32_t tid = threadIdx.x;
    __shared__ int shared[BLOCK_SIZE];

    shared[tid] = arr[tid];
    __syncthreads();

    for (uint32_t k = 2; k <= BLOCK_SIZE; k *= 2) {
        for (uint32_t j = k / 2; j > 0; j /= 2) {
            uint32_t ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (shared[tid] > shared[ixj]) {
                        SWAP(shared[tid], shared[ixj]);
                    }
                } else {
                    if (shared[tid] < shared[ixj]) {
                        SWAP(shared[tid], shared[ixj]);
                    }
                }
            printf("k=%d j=%d tid=%d ixj=%d %d\n", k, j, tid, ixj, ((tid & k) == 0));
            }
            __syncthreads();
        }
    }

    arr[tid] = shared[tid];
}

void bitonic_sort(int* arr, const uint32_t size)
{
    int* dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(int) * size));
    CSC(cudaMemcpy(dev_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice));
    for (uint32_t block_start = 0; block_start + BLOCK_SIZE < size; block_start += BLOCK_SIZE) {
        START_KERNEL((kernel_bitonic_sort<<<1, BLOCK_SIZE, sizeof(int) * BLOCK_SIZE>>>(dev_arr + block_start)));
    }
    CSC(cudaMemcpy(arr, dev_arr, sizeof(int) * size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_arr));
}

void odd_even_sorting(int* arr, const uint32_t size)
{
    for (size_t i = 0; i < size; i++) {
        for (size_t j = i & 1; j < size - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                SWAP(arr[j], arr[j + 1]);
            }
        }
    }
}

void odd_even(int* arr, const uint32_t size)
{
}

int main()
{
    uint32_t size = scan_4();

    int* arr = (int*)malloc(size * sizeof(int));
    for (uint32_t i = 0; i < size; ++i) {
        arr[i] = int(scan_4());
    }

    bitonic_sort(arr, size);
    // odd_even_sorting(arr, size);

    print_arr(stdout, arr, size);

    free(arr);
    return 0;
}