#include <cmath>
#include <iostream>

#include "helpers.cuh"

#include <time.h>

// call this function to start a nanosecond-resolution timer
struct timespec timer_start()
{
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

typedef unsigned long long ull;

// call this function to end a timer, returning nanoseconds elapsed as a long
ull timer_end(struct timespec start_time)
{
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    ull diffInNanos = (ull)(end_time.tv_sec - start_time.tv_sec) * (ull)1e9 + (ull)(end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}

int main()
{
    int n, m;
    std::cin >> n >> m;

    if (n == 0 || m == 0) {
        return 0;
    }

    host_matrix A(n * m);
    read_matrix(A, n, m);

    struct timespec start_time = timer_start();
    host_matrix res = transponse(A, n, m);
    fprintf(stderr, "time: %lld\n", timer_end(start_time));

    show_matrix(stdout, res, m, n);
    return 0;
}