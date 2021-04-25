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

#define __bcast(val, size, type) MPI_ERR(MPI_Bcast(val, size, type, ROOT_RANK, MPI_COMM_WORLD))
#define bcast_bytes(val, size) __bcast(val, size, MPI_BYTE)
#define bcast_int(val) __bcast(val, 1, MPI_INT)
#define bcast_float(val) __bcast(val, 1, MPI_FLOAT)
#define bcast_vec3(val) __bcast(val, 3, MPI_FLOAT)

#endif
