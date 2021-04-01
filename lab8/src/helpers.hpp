#ifndef HELPERS_HPP
#define HELPERS_HPP

#define ROOT_RANK 0

#define MPI_ERR(call)                                            \
    do {                                                         \
        int err = (call);                                        \
        if (err != MPI_SUCCESS) {                                \
            char estring[MPI_MAX_ERROR_STRING];                  \
            int len;                                             \
            MPI_Error_string(err, estring, &len);                \
            fprintf(stderr, "MPI ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, estring);                    \
            MPI_Finalize();                                      \
            exit(0);                                             \
        }                                                        \
    } while (false)

#define bcast(val, type) MPI_ERR(MPI_Bcast(val, 1, type, ROOT_RANK, MPI_COMM_WORLD))
#define bcast_int(val) bcast(val, MPI_INT)
#define bcast_double(val) bcast(val, MPI_DOUBLE)

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

#endif
