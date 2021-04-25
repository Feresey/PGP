#include "helpers.cuh"

void setDevice(int device_id)
{
    CUDA_ERR(cudaSetDevice(device_id));
}

void getDeviceCount(int* device_count)
{
    CUDA_ERR(cudaGetDeviceCount(device_count));
}
