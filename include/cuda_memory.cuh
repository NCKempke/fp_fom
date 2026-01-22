#include <cuda_runtime.h>

template<typename T>
T *copyinit_to_device(const std::vector<T> &host_vec)
{
    T *device_ptr;
    const auto n_bytes_alloc = host_vec.size() * sizeof(T);
    cudaMalloc(&device_ptr, n_bytes_alloc);
    cudaMemcpy(device_ptr, host_vec.data(), n_bytes_alloc, cudaMemcpyHostToDevice);

    return device_ptr;
}

template<typename T>
void copy_from_device(std::vector<T> &host_vec, const T* device_ptr, size_t n_elem) {
    const auto n_bytes_alloc = n_elem * sizeof(T);
    cudaMemcpy(host_vec.data(), device_ptr, n_bytes_alloc, cudaMemcpyDeviceToHost);
}

template<typename T>
void device_free(T* device_prt) {
    cudaFree(device_prt);
}
