#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <mpi.h>

#include <fstream>
#include <string>

#define ROOT_RANK 0

#define CSC(call)                                            \
    do {                                                     \
        int err = (call);                                    \
        if (err != MPI_SUCCESS) {                            \
            char estring[MPI_MAX_ERROR_STRING];              \
            int len;                                         \
            MPI_Error_string(err, estring, &len);            \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, estring);                \
            MPI_Finalize();                                  \
            exit(0);                                         \
        }                                                    \
    } while (false)

template <class T>
class dim3 {
    struct dim3_print {
        const std::string& prefix;
        const dim3<T>& data;
        dim3_print(const dim3<T>& data);
        friend std::ostream& operator<<(std::ostream& out, const dim3_print& data);
    };

public:
    T x, y, z;

    dim3(T x = 0, T y = 0, T z = 0);
    dim3(std::istream& in);
    friend std::istream& operator>>(std::istream& in, dim3<T>& val);
    dim3_print print(const std::string& prefix);
};

class Solver {
    int rank;
    int n_processes;

    dim3<int> n_blocks;
    dim3<int> block_size;
    double eps;
    dim3<double> l_size;
    double u_down, u_up, u_left, u_right, u_front, u_back;
    double u_0;

    std::string output_file_path;

    void read_data(std::istream& in);
    void show_data(std::ostream& out);
    void mpi_bcast();

public:
    Solver(std::istream& in);
};

#endif