#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <type_traits>

constexpr double FEASTOL = 1e-6;
constexpr double EPSILON = 1e-9;
constexpr int WARP_SIZE = 32;
constexpr int WARP_SIZE_HALF = WARP_SIZE / 2;

#define FULL_WARP_MASK 0xffffffff

#define assert_iff(prop1, prop2) (assert((prop1) == (prop2)))
#define assert_if_then(antecedent, consequent) (assert(!(antecedent) || (consequent)))
#define assert_if_then_else(cond, then_expr, else_expr) (assert((!(cond) || (then_expr)) && ((cond) || (else_expr))))


#ifndef NDEBUG

#define CHECK_CUDA(...)                                                                                \
    {                                                                                                  \
        (__VA_ARGS__);                                                                                 \
        cudaError_t status = cudaGetLastError();                                                       \
        if (status != cudaSuccess)                                                                     \
        {                                                                                              \
            printf("[%s:%d] CUDA Runtime failed with error %d: %s\n", __FILE__, __LINE__, (int)status, \
                   cudaGetErrorString(status));                                                        \
        }                                                                                              \
    }

#else
#define CHECK_CUDA(...)                                                                                \
    {                                                                                                  \
        (__VA_ARGS__);                                                                                 \
    }
#endif

// functions for comparisons with absolute tolerance only
__device__ __host__ bool is_zero(double a)
{
    return abs(a) <= EPSILON;
}

__device__ __host__ bool is_eq(double a, double b)
{
    return abs(a - b) <= EPSILON;
}

__device__ __host__ bool is_integer(double a)
{
    return is_eq(round(a), a);
}

__device__ __host__ bool is_ge(double a, double b)
{
    // a >= b considering tolerance
    return a - b >= -EPSILON;
}

__device__ __host__ bool is_gt(double a, double b)
{
    // a >= b considering tolerance
    return a - b > EPSILON;
}

__device__ __host__ bool is_le(double a, double b)
{
    // a <= b considering tolerance
    return a - b <= EPSILON;
}

__device__ __host__ bool is_lt(double a, double b)
{
    // a <= b considering tolerance
    return a - b < -EPSILON;
}

__device__ __host__ bool is_zero_feas(double a)
{
    return abs(a) <= FEASTOL;
}

__device__ __host__ bool is_eq_feas(double a, double b)
{
    return abs(a - b) <= FEASTOL;
}

__device__ __host__ bool is_ge_feas(double a, double b)
{
    // a >= b considering tolerance
    return a - b >= -FEASTOL;
}

__device__ __host__ bool is_le_feas(double a, double b)
{
    // a <= b considering tolerance
    return a - b <= FEASTOL;
}

__device__ __host__ bool is_gt_feas(double a, double b)
{
    // a >= b considering tolerance
    return a - b > FEASTOL;
}

__device__ __host__ bool is_lt_feas(double a, double b)
{
    // a <= b considering tolerance
    return a - b < -FEASTOL;
}

/* TODO: reduction operation should be a template, too. */

/* Compute min(val) within the warp. Assumes that full warp participates. */
template <typename VAL_TYPE> __inline__ __device__ VAL_TYPE warp_min_reduce(VAL_TYPE val)
{
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(FULL_WARP_MASK, (unsigned)val);
#else
  /* Fallback: implement warp-level min reduction manually. */
  for (int offset = WARP_SIZE_HALF; offset > 0; offset /= 2)
    val = min(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));

  /* Broadcast back the result from lane 0. */
  return __shfl_sync(FULL_WARP_MASK, val, 0);
#endif
}

/* Compute min(val) within the warp. Assumes that full warp participates. */
template <typename VAL_TYPE> __inline__ __device__ VAL_TYPE warp_sum_reduce(VAL_TYPE val)
{
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(FULL_WARP_MASK, (unsigned)val);
#else
  /* Fallback: implement warp-level min reduction manually. */
  for (int offset = WARP_SIZE_HALF; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_WARP_MASK, val, offset);

  /* Broadcast back the result from lane 0. */
  return __shfl_sync(FULL_WARP_MASK, val, 0);
#endif
}

/** Initialize the random state for the whole block, given a seed. Should only be called from thread 0. */
template <const int WARPS_PER_BLOCK>
__device__ void init_curand_warp(curandState &state, size_t seed)
{
    const int warp_id = threadIdx.x / WARP_SIZE;

    curand_init(seed, blockIdx.x * WARPS_PER_BLOCK + warp_id, 0, &state);
}

/** Returns, for the whole block, a random double in [beg, end]. */
__device__ double get_random_double_in_range_warp(curandState &state, double beg, double end)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    double randval = 0.0;

    if (thread_idx_warp == 0)
        randval = beg + curand_uniform_double(&state) * (end - beg);

    randval = __shfl_sync(FULL_WARP_MASK, randval, 0);

    return randval;
}

/** Returns, for the whole block, a random integer between [0,..,n) excluding n. State is this block's curand state. */
__device__ int get_random_int_warp(curandState &state, int n)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    int randval;

    if (thread_idx_warp == 0)
    {
        if (n > 0) {
            unsigned int r = curand(&state);
            randval = r % n;
        } else {
            randval = 0;
        }
    }
    randval = __shfl_sync(FULL_WARP_MASK, randval, 0);

    return randval;
}

/** Returns, for the calling thread and its random state, a random integer between [0,..,n) excluding n. */
__device__ int get_random_int_thread(curandState &state, int n) {
    int randval;

    if (n > 0) {
        unsigned int r = curand(&state);
        randval = r % n;
    } else {
        randval = 0;
    }
    return randval;
}

/* Templated wrappers for array copying. */
template<typename T>
cudaError_t cudaMemcpyAuto(T* dst, const T* src, size_t count) {
    return cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDefault);
}

template<typename T>
cudaError_t cudaMemcpyAutoAsync(T* dst, const T* src, size_t count,
                               const cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDefault, stream);
}

/* Templated wrappers for single elements. */
template<typename T>
cudaError_t cudaMemcpyAuto(T* dst, const T* src) {
    return cudaMemcpy(dst, src, sizeof(T), cudaMemcpyDefault);
}

template<typename T>
cudaError_t cudaMemcpyAutoAsync(T* dst, const T* src,
                               const cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, sizeof(T), cudaMemcpyDefault, stream);
}

template <typename T>
inline void copy_host_to_device(const std::vector<T>& h,
                           thrust::device_vector<T>& d,
                           cudaStream_t stream)
{
    assert(d.size() >= h.size());
    const size_t bytes = h.size() * sizeof(T);

    CHECK_CUDA(cudaMemcpyAutoAsync(thrust::raw_pointer_cast(d.data()), h.data(), h.size(), stream));
}

template <typename T>
inline void copy_device_to_host(const thrust::device_vector<T>& d,
                           std::vector<T>& h,
                           cudaStream_t stream)
{
    assert(h.size() >= d.size());

    CHECK_CUDA(cudaMemcpyAutoAsync(h.data(), thrust::raw_pointer_cast(d.data()), d.size(), stream));
}
