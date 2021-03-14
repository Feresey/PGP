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

#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#define __shared__
#endif

#endif
