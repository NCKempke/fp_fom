#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr double FEASTOL = 1e-6;
constexpr double EPSILON = 1e-9;

#define assert_iff(prop1, prop2) (assert((prop1) == (prop2)))
#define assert_if_then(antecedent, consequent) (assert(!(antecedent) || (consequent)))
#define assert_if_then_else(cond, then_expr, else_expr) (assert((!(cond) || (then_expr)) && ((cond) || (else_expr))))

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

/** Initialize the random state for the whole block, given a seed. Should only be called from thread 0. */
__device__ void init_curand_block(curandState &state, size_t seed)
{
    curand_init(seed, blockIdx.x, 0, &state);
}

/** Returns, for the whole block, a random double in [beg, end]. */
__device__ double get_random_double_in_range_block(curandState &state, double beg, double end)
{
    __shared__ double randval;

    if (threadIdx.x == 0)
        randval = beg + curand_uniform_double(&state) * (end - beg);
    __syncthreads();

    return randval;
}

/** Returns, for the whole block, a random integer between [0,..,n) excluding n. State is this block's curand state. */
__device__ int get_random_int_block(curandState &state, int n)
{
    __shared__ int randval;

    if (threadIdx.x == 0)
    {
        if (n > 0) {
            unsigned int r = curand(&state);
            randval = r % n;
        } else {
            randval = 0;
        }
    }
    __syncthreads();

    assert(0 <= randval);
    if (n != 0)
        assert(randval < n);

    return randval;
}
