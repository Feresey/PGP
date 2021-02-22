#include <algorithm>
#include <assert.h>
#include <cstring>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.cuh"

#define BLOCK_SIZE 1024

// попытка выровнять блоки
#define THREAD_INDEX(idx) ((WARP_SIZE + 1) * ((idx) / WARP_SIZE) + ((idx) % WARP_SIZE))

#define SWAP_IF(arr, x, y) \
    if (arr[x] > arr[y])   \
    SWAP(arr, x, y)

void odd_even_sort(int* x, const uint32_t n)
{
    uint32_t sort = 0, i;
    while (!sort) {
        sort = 1;
        for (i = 1; i < n - 1; i += 2) {
            if (x[i] > x[i + 1]) {
                std::swap(x[i], x[i + 1]);
                sort = 0;
            }
        }
        for (i = 0; i < n - 1; i += 2) {
            if (x[i] > x[i + 1]) {
                std::swap(x[i], x[i + 1]);
                sort = 0;
            }
        }
    }
}

void bitonicSeqMerge(int a[], int start, int BseqSize, int direction)
{
    if (BseqSize > 1) {
        int k = BseqSize / 2;
        for (int i = start; i < start + k; i++) {
            if (direction == (a[i] > a[i + k])) {
                std::swap(a[i], a[i + k]);
            }
        }
        bitonicSeqMerge(a, start, k, direction);
        bitonicSeqMerge(a, start + k, k, direction);
    }
}
void bitonicSortrec(int a[], int start, int BseqSize, int direction)
{
    if (BseqSize > 1) {
        int k = BseqSize / 2;
        bitonicSortrec(a, start, k, 1);
        bitonicSortrec(a, start + k, k, 0);
        bitonicSeqMerge(a, start, BseqSize, direction);
    }
}

void bitonicSort(int a[], int size, int up)
{
    bitonicSortrec(a, 0, size, up);
}

uint32_t nearest_size(const uint32_t num, const uint32_t prod)
{
    uint32_t mod = num % prod;
    if (mod == 0) {
        return num;
    }
    return num + (prod - mod);
}

void sort(int* arr, const uint32_t n)
{
    // размер, кратный размеру блока
    const uint32_t dev_n = nearest_size(n, BLOCK_SIZE);
    const uint32_t n_blocks = dev_n / BLOCK_SIZE;

    int* dev_arr = (int*)malloc(dev_n * sizeof(int));
    for (uint32_t i = 0; i < n; i++) {
        dev_arr[i] = arr[i];
    }
    for (uint32_t i = n; i < dev_n; i++) {
        dev_arr[i] = INT_MIN;
    }

    // предварительная сортировка блоков
    for (uint32_t start = 0; start < dev_n; start += BLOCK_SIZE) {
        odd_even_sort(dev_arr + start, BLOCK_SIZE);
    }

    // ну если массив влез в 1 блок то зачем его сортировать ещё раз?
    if (n_blocks == 1) {
        goto END;
    }

    for (uint32_t iter = 0; iter < n_blocks; ++iter) {
        for (uint32_t start = BLOCK_SIZE / 2; start < dev_n - BLOCK_SIZE; start += BLOCK_SIZE) {
            bitonicSort(dev_arr + start, BLOCK_SIZE, 1);
        }
        for (uint32_t start = 0; start < dev_n; start += BLOCK_SIZE) {
            bitonicSort(dev_arr + start, BLOCK_SIZE, 1);
        }
    }

END:
    memcpy(arr, dev_arr + (dev_n - n), n * sizeof(int));
    free(dev_arr);
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
    if (size == 0) { // foolproof
        return 0;
    }

    int* arr = (int*)malloc(size * sizeof(int));
    for (uint32_t i = 0; i < size; ++i) {
        arr[i] = int(scan_4());
    }

    struct timespec start_time = timer_start();
    sort(arr, size);
    fprintf(stderr, "sort time: %ld\n", timer_end(start_time));
    fwrite(arr, sizeof(int), size, stdout);

    free(arr);
    return 0;
}