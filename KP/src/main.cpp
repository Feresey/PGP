#include <chrono>
#include <ctime>
#include <mpi.h>

#include "helpers.cuh"
#include "helpers.hpp"

#include "render/render.cuh"
#include "scene.hpp"

void show_best()
{
    std::cout
        << "1024\n"
        << "%d.data\n"
        << "320 240 120\n"
        << "7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0\n"
        << "2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n"
        << "2.0 0.0 0.0 1.0 0.0 0.0 1.0 0.9 0.1 10\n"
        << "0.0 2.0 0.0 0.0 1.0 0.0 0.75 0.8 0.2 5\n"
        << "0.0 0.0 0.0 0.0 0.7 0.7 0.5 0.7 0.3 2\n"
        << "-5.0 -5.0 -1.0 -5.0 5.0 -1.0 5.0 5.0 -1.0 5.0 -5.0 -1.0 "
        << "floor.data 0.0 1.0 0.0 0.5\n"
        << "2\n"
        << "-10.0 0.0 10.0 1.0 1.0 1.0\n"
        << "1.0 0.0 10.0 0.0 0.0 1.0\n"
        << "10 16\n";
}

void write_image(
    const std::string& path,
    const uchar4* data, int w, int h)
{
    MPI_File file;
    MPI_ERR(MPI_File_open(MPI_COMM_WORLD, path.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file));
    MPI_ERR(MPI_File_write(file, &w, 1, MPI_INT, MPI_STATUS_IGNORE));
    MPI_ERR(MPI_File_write(file, &h, 1, MPI_INT, MPI_STATUS_IGNORE));
    MPI_ERR(MPI_File_write(file, data, int(sizeof(uchar4)) * w * h, MPI_BYTE, MPI_STATUS_IGNORE));
    MPI_ERR(MPI_File_close(&file));
}

int main(int argc, char* argv[])
{
    MPI_ERR(MPI_Init(&argc, &argv));
    int rank, n_processes;
    MPI_ERR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_ERR(MPI_Comm_size(MPI_COMM_WORLD, &n_processes));
    MPI_ERR(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    ComputeMode mode = CUDA;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--default") {
            show_best();
            MPI_ERR(MPI_Finalize());
            return 0;
        }
        if (std::string(argv[i]) == "--cpu") {
            mode = OPEN_MP;
            break;
        }
        if (std::string(argv[i]) == "--gpu") {
            mode = CUDA;
            break;
        }
    }

    if (mode == CUDA) {
        MPI_Comm local_comm;
        int local_rank;
        MPI_ERR(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        MPI_ERR(MPI_Comm_rank(local_comm, &local_rank));
        MPI_ERR(MPI_Comm_free(&local_comm));

        int n_devices;
        getDeviceCount(&n_devices);
        setDevice(local_rank % n_devices);
    }

    Scene scene;
    if (rank == 0) {
        debug("here");
        std::cin >> scene;
        std::cerr << scene << std::endl;
    }
    scene.mpi_bcast();

    Renderer* renderer = NewRenderer(mode, scene);

    // if (rank == 0) {
    //     // import_obj_to_scene(scene_trigs, "hex.obj", task.hex);
    //     // import_obj_to_scene(scene_trigs, "octa.obj", task.octa);
    //     // import_obj_to_scene(scene_trigs, "icos.obj", task.icos);
    //     // add_floor_to_scene(scene_trigs, task.floor);
    // }
    // renderer->mpi_bcast_poly();

    MPI_ERR(MPI_Barrier(MPI_COMM_WORLD));

    for (int frame = rank; frame < scene.n_frames; frame += n_processes) {
        auto start = std::chrono::high_resolution_clock::now();
        renderer->Render(frame);
        auto end = std::chrono::high_resolution_clock::now();

        char output_path[256];
        sprintf(output_path, scene.output_pattern.data(), frame);
        write_image(output_path, renderer->data(), scene.w, scene.h);

        std::cerr
            << frame << "\t"
            << output_path << "\t"
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
            << std::endl;
    }

    delete renderer;
    MPI_ERR(MPI_Finalize());
    return 0;
}
