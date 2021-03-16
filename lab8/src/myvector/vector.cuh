#ifndef MY_VECTOR_CUH
#define MY_VECTOR_CUH

#include <vector>

#include "helpers.cuh"

template<class T>
struct DeviceVector {
    std::vector<T>& data;
    T* device_data;

    DeviceVector(std::vector<T>& data):
    data(data){
        CUDA_ERR(cudaMalloc(&this->device_data, data.size()*sizeof(T)));
        memcpy_to_device();
    }
    void memcpy_to_device() {
        CUDA_ERR(cudaMemcpy(this->device_data, data.data(), data.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
    void memcpy_to_host() {
        CUDA_ERR(cudaMemcpy(data.data(), this->device_data, data.size()*sizeof(T), cudaMemcpyDeviceToHost));
    }
    ~DeviceVector() {
        memcpy_to_host();
        CUDA_ERR(cudaFree(this->device_data));
    }
};

#endif
