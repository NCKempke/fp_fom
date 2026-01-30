#include "evolution_search.cuh"

#include "gpu_data.cuh"
#include "mip.h"

#include <consolelog.h>

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

#include "cub/cub.cuh"
#include <cub/util_device.cuh>
#include <cub/device/device_select.cuh>

constexpr int N_MAX_MOVE_BLOCKS = 512; /* Maximum number of blocks used for any move kernel */
constexpr int BLOCKSIZE_VECTOR_KERNEL = 1024; /* Blocksize used for vector kernels (each thread operating on one vector element). */

constexpr int N_BLOCKS_SINGLE_COL_MOVE = 512;
constexpr int BLOCKSIZE_SINGLE_COL_MOVE = 32;
constexpr int AVAILABLE_MOVED = 5;

/* Moves:
 * - one_opt (feas)   : implemented not applied
 * - one_opt (greedy) : push variable in direction of its objective
 * - flip             : flips a binary randomly selected variable
 * - random           : selects a random variable and assigns it a random value
 * TODO:
 * - mtm_satisfied    : select a random satisfied constraint and set slack to zero
 * - mtm_unsatisfied  : selects a random violated constraint, then selects a variable within its range
 *                      and adjusts it to make the constraint as feasible as possible
 *
 * TODO:
 * - swap             : select two (binary) variables with different values and swap them
 * - Lagrange         : from Feaspump
 *
 * - TSP swap?
 * - Avoid duplicate moves.
 *
 * - Solution pool;
 * - Sync solutions from FPR;
 * - Scoring function;
 * - Apply "all" for "small" problems instead of random
 */

constexpr double FEASTOL = 1e-6;
constexpr double EPSILON = 1e-9;

struct single_col_move
{
    double val;
    int col;
};

struct swap_move
{
    int col1;
    int col2;
};


/* Large scores are bad. */
struct solution_score
{
    double objective = DBL_MAX;
    double violation = DBL_MAX;

    __host__ __device__ inline double feas_score() const
    {
        // TODO
        return /*objective +*/ violation;
    }
};

// TODO : move to utils header
#define assert_iff(prop1, prop2)               (assert((prop1) == (prop2)))
#define assert_if_then(antecedent, consequent) (assert(!(antecedent) || (consequent)))
#define assert_if_then_else(cond, then_expr, else_expr)   (assert((!(cond) || (then_expr)) && ((cond) || (else_expr))))

// functions for comparisons with absolute tolerance only
__device__ __host__ bool is_zero(double a) {
    return abs(a) <= EPSILON;
}

__device__ __host__ bool is_eq(double a, double b) {
    return abs(a - b) <= EPSILON;
}

__device__ __host__ bool is_ge(double a, double b) {
    // a >= b considering tolerance
    return a-b >= -EPSILON;
}

__device__ __host__ bool is_gt(double a, double b) {
    // a >= b considering tolerance
    return a-b > EPSILON;
}

__device__ __host__ bool is_le(double a, double b) {
    // a <= b considering tolerance
    return a-b <= EPSILON;
}

__device__ __host__ bool is_lt(double a, double b) {
    // a <= b considering tolerance
    return a-b < -EPSILON;
}

__device__ __host__ bool is_zero_feas(double a) {
    return abs(a) <= FEASTOL;
}

__device__ __host__ bool is_eq_feas(double a, double b) {
    return abs(a - b) <= FEASTOL;
}

__device__  __host__ bool is_ge_feas(double a, double b) {
    // a >= b considering tolerance
    return a-b >= -FEASTOL;
}

__device__  __host__ bool is_le_feas(double a, double b) {
    // a <= b considering tolerance
    return a-b <= FEASTOL;
}

__device__ __host__ bool is_gt_feas(double a, double b) {
    // a >= b considering tolerance
    return a-b > FEASTOL;
}

__device__ __host__ bool is_lt_feas(double a, double b) {
    // a <= b considering tolerance
    return a-b < -FEASTOL;
}

__global__ void init_rng_per_block(curandState *states, unsigned long seed)
{
    int block_idx = blockIdx.x;

    if (block_idx >= N_MAX_MOVE_BLOCKS)
        return;

    curand_init(seed, block_idx, 0, &states[block_idx]);
}

/* Returns for threadIdx.x == 0 the score after virtually applying give move. */
__device__ solution_score compute_score_single_col_move(const GpuModelPtrs &model, const double *slack, const double *sol, single_col_move move, double objective, double sum_slack)
{
    const int thread_idx = threadIdx.x;
    const int col = move.col;

    const double obj_coef = model.objective[col];
    const double col_val = sol[col];
    const double delta = move.val - col_val;
    const double delta_obj = delta * obj_coef;
    double slack_change_thread = 0.0;

    assert(model.lb[col] <= move.val && move.val <= model.ub[col]);
    // if (thread_idx == 0)
    // {
    //     printf("fixval: %g; colval: %g\n", move.val, col_val);
    // }

    //TODO: is this correct to skip here?
    if (is_eq(delta, 0))
        return{objective, sum_slack};
    /* Iterate column and compute changes in violation. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* We have <= and = only. */
        const int is_eq = (model.row_sense[row_idx] == 'E');

        const double slack_old = slack[row_idx];
        const double slack_new = slack_old - coef * delta;

        double viol_old = is_eq * abs(slack_old) + (1 - is_eq) * max(0.0, -slack_old);
        double viol_new = is_eq * abs(slack_new) + (1 - is_eq) * max(0.0, -slack_new);

        slack_change_thread += (viol_new - viol_old);
    }

    using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_SINGLE_COL_MOVE>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ typename BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    const double slack_change = BlockReduce(temp_storage).Sum(slack_change_thread);

    return {objective + delta_obj, sum_slack + slack_change};
}

/* Returns for threadIdx.x == 0 the score after virtually applying give move. */
__device__ solution_score compute_score_col_swap(const GpuModelPtrs &model, const double *slack, const double *sol, swap_move move, double objective, double sum_slack)
{
    const int thread_idx = threadIdx.x;
    const int col1 = move.col1;
    const int col2 = move.col2;

    const double obj_coef1 = model.objective[col1];
    const double obj_coef2 = model.objective[col2];
    const double col_val1 = sol[col1];
    const double col_val2 = sol[col2];
    const double delta1 = col_val2 - col_val1;
    const double delta2 = col_val1 - col_val2;
    const double delta_obj = delta1 * obj_coef1 + delta2 * obj_coef2;
    double slack_change_thread = 0.0;

    // if (thread_idx == 0)
    // {
    //     printf("fixval: %g; colval: %g\n", move.val, col_val);
    // }

    //TODO: is this correct to skip here?
    if (is_eq(delta1, 0)) {
        assert(delta2 == 0);
        return{objective, sum_slack};
    }
    /* Iterate column and compute changes in violation. */

    //TODO: write function here to reduce duplicates
    const int col_beg = model.row_ptr_trans[col1];
    const int col_end = model.row_ptr_trans[col1 + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* We have <= and = only. */
        const int is_eq = (model.row_sense[row_idx] == 'E');

        const double slack_old = slack[row_idx];
        const double slack_new = slack_old - coef * delta1;

        double viol_old = is_eq * abs(slack_old) + (1 - is_eq) * max(0.0, -slack_old);
        double viol_new = is_eq * abs(slack_new) + (1 - is_eq) * max(0.0, -slack_new);

        slack_change_thread += (viol_new - viol_old);
    }

    const int col_beg_2 = model.row_ptr_trans[col2];
    const int col_end_2 = model.row_ptr_trans[col2 + 1];

    for (int inz = col_beg_2 + thread_idx; inz < col_end_2; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* We have <= and = only. */
        const int is_eq = (model.row_sense[row_idx] == 'E');

        const double slack_old = slack[row_idx];
        const double slack_new = slack_old - coef * delta2;

        double viol_old = is_eq * abs(slack_old) + (1 - is_eq) * max(0.0, -slack_old);
        double viol_new = is_eq * abs(slack_new) + (1 - is_eq) * max(0.0, -slack_new);

        slack_change_thread += (viol_new - viol_old);
    }


    using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_SINGLE_COL_MOVE>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ typename BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    const double slack_change = BlockReduce(temp_storage).Sum(slack_change_thread);

    return {objective + delta_obj, sum_slack + slack_change};
}

/* activities = rhs - Ax */
__device__ void compute_random_move(const GpuModelPtrs &model, curandState &state, const double *slack, const double *sol, double objective, double sum_slack, int col, solution_score &best_score, single_col_move &best_move, int ncols)
{
    const int thread_idx = threadIdx.x;

    if (col >= ncols)
        return;

    // TODO column must be integer
    double lb = model.lb[col];
    double ub = model.ub[col];

    __shared__ double random_val;
    if (thread_idx == 0)
    {
        random_val = curand_uniform(&state);
    }
    /* Make sure shared memory is visible to all threads in the block. */
    __syncthreads();

    /* Don't run if these are too close. */
    if (ub - lb < 0.001)
        return;

    // TODO: fixval != col_val
    double fixval = lb + (ub - lb) * random_val;

    if (model.var_type[col] == 'I')
        fixval = static_cast<int>(fixval + 0.5);
    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move(model, slack, sol, {fixval, col}, objective, sum_slack);

    /* Write violation to global memory. */
    if (thread_idx == 0)
    {
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fixval, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* activities = rhs - Ax */
template <const bool GREEDY>
__device__ void compute_oneopt_move(const GpuModelPtrs &model, curandState &state, const double *slack, const double *sol, double objective, double sum_slack, int col, solution_score &best_score, single_col_move &best_move, int ncols)
{
    const int thread_idx = threadIdx.x;

    if (col >= ncols)
        return;

    // TODO column must be integer ?
    double col_val = sol[col];
    double lb = model.lb[col];
    double ub = model.ub[col];
    double obj = model.objective[col];

    /* Only do this for variables with non-zero objective. */
    if ((is_gt(obj, 0.0) && is_eq(col_val, lb)) || (is_lt(obj, 0.0) && is_eq(col_val, ub)) || is_eq(obj, 0.0))
        return;

    assert(lb <= col_val && col_val <= ub);

    /* Positive stepsize in objective direction. */
    double stepsize = DBL_MAX;

    /* Get minimum slack in locking direction. Iterate column i and for each row j that locks in objective direction and compute min_j(row_coeff_ij * slack_j). */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    if (GREEDY)
    {
        stepsize = obj > 0.0 ? col_val - lb : ub - col_val;
    }
    else
    {
        for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
        {
            const double coef = model.row_val_trans[inz];
            const int row_idx = model.col_idx_trans[inz];
            const char sense = model.row_sense[row_idx];
            const char row_slack = slack[row_idx];
            const int is_eq = (sense == 'E');
            const int is_objcoef_pos = (obj * coef > 0.0);
            assert(row_slack >= 0.0);

            const double scaled_slack = fabs(coef) * row_slack;

            // TODO : maybe allow infeasible solutions here and allow becoming infeasible
            //TODO: rounding is not considered yet
            stepsize = min(stepsize, (1 - is_eq) * (is_objcoef_pos * stepsize + (1 - is_objcoef_pos) * scaled_slack));
            // if ((sense == 'L' && obj * coef < 0.0)) {
            //     if (coef > 0.0) {
            //         stepsize = min(stepsize, coef * row_slack);
            //     } else {
            //         /* ax - b y <= rhs */
            //         stepsize  = min(stepsize, -coef * row_slack);
            //     }
            // }
            //  else if (sense == 'E') {
            //     stepsize = 0.0;
            // }
        }

        /* Reduce min of stepsize . */
        using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_SINGLE_COL_MOVE>;

        /* Allocate shared memory for BlockReduce. */
        __shared__ typename BlockReduce::TempStorage temp_storage;

        /* Reduce all slack changes to thread 0 of this block. */
        stepsize = BlockReduce(temp_storage).Reduce(stepsize, cuda::minimum());
    }

    assert(stepsize >= 0.0);

    if (stepsize == 0.0)
        return;

    const double fixval = obj > 0.0 ? col_val - stepsize : col_val + stepsize;
    assert(lb <= fixval && fixval <= ub);

    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move(model, slack, sol, {fixval, col}, objective, sum_slack);

    /* Write violation to global memory. */
    if (thread_idx == 0)
    {
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fixval, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}


__device__ void compute_flip_move(const GpuModelPtrs &model, const double *slack, const double *sol, const double objective, const double sum_slack, const int col, solution_score
                                  &best_score, single_col_move &best_move, int ncols)
{
    const int thread_idx = threadIdx.x;

    if (col >= ncols)
        return;

    double lb = model.lb[col];
    double ub = model.ub[col];

    // is binary variable?
    bool active = (model.var_type[col] == 'I' && lb == 0 && ub == 1);
    double fixval;
    solution_score score;
    if (active) {
        fixval = sol[col] > 0.5 ? 0 : 1;

        /* score is valid only for threadIdx.x == 0 */
        score = compute_score_single_col_move(model, slack, sol, {fixval, col}, objective, sum_slack);
    }
    /* Write violation to global memory. */
    if (thread_idx == 0 && active)
    {
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fixval, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* Compute the mtm move for an unsatisfied constraint. */
__device__ void compute_mtm_unsat_move(const GpuModelPtrs &model, const double *slack, const double *sol, const double objective, const double sum_slack, const int row, const int col_index, solution_score
                                  &best_score, single_col_move &best_move, const int nrows)
{
    assert(row < nrows);
    assert(col_index < (model.row_ptr[row + 1] - model.row_ptr[row]));

    const double slack_for_row = slack[row];
    const bool is_row_eq = model.row_sense[row] == 'E';
    const bool slack_is_pos = is_gt_feas(slack_for_row, 0);

    /* skip feasible constraints -> either equation with slack == 0 or inequalities with positive slack */
    const bool is_row_feas = ((is_row_eq && is_zero_feas(slack_for_row)) || slack_is_pos);

    if (is_row_feas)
        return;

    const int col = model.col_idx[model.row_ptr[row] + col_index];
    double coeff = model.row_val[model.row_ptr[row] + col_index];

    const double lb = model.lb[col];
    const double ub = model.ub[col];
    const double old_val = sol[col];

    solution_score score;
    double fix_val;
    bool move_up;

    /* Try to move col as far as possible to make the constraint exactly tight/feasible; as we know the row is infeasible, move the slack rhs - a'x towards zero. */

    if (slack_is_pos) {
        move_up = coeff > 0.0 ? true : false;
    } else {
        move_up = coeff > 0.0 ? false : true;
    }

    /* Skip if colum is at already at the bound we want to move it towards. */
    if ((move_up && is_eq(ub, old_val)) || (is_eq(lb, old_val)))
        return;

    assert(coeff != 0.0);

    /* Exact value that makes slack zero; we need
     *      rhs - a'x - aj dxj == 0
     * <=>  rhs - a'x           = aj dxj
     * <=> (rhs - a'x) / aj     = dxj
     */
    fix_val = old_val + slack_for_row / coeff;
    assert_if_then_else(move_up, fix_val > old_val, fix_val < old_val);

    if (model.var_type[col] != 'C')
        fix_val = static_cast<int>(fix_val + move_up);

    fix_val = move_up ? fmin(fix_val, ub) : fmax(fix_val, lb);
    assert(lb <= fix_val && fix_val <= ub);

    // printf("row %d col %d old %d  new %d %d\n", row, col,  old_val, fix_val);

    /* score is valid only for threadIdx.x == 0 */
    score = compute_score_single_col_move(model, slack, sol, {fix_val, col}, objective, sum_slack);

    /* Write violation to global memory. */
    if (threadIdx.x == 0)
    {
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fix_val, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

__device__ void compute_mtm_sat_move(const GpuModelPtrs &model, const double *slack, const double *sol, const double objective, const double sum_slack, const int row, const int col_index, solution_score
                                  &best_score, single_col_move &best_move, const int nrows)
{
    const int thread_idx = threadIdx.x;

    if (row >= nrows)
        return;

    const double slack_for_row = slack[row];
    // skip equations or unsatisfied inequalities (slack < 0)
    bool active =!( model.row_sense[row] != 'E' && is_le(slack_for_row, 0));

    solution_score score;
    double fixval;
    const int col = model.col_idx[model.row_ptr[row] + col_index];
    double coeff = model.row_val[model.row_ptr[row] + col_index];
    if (active) {
        const double lb = model.lb[col];
        const double ub = model.ub[col];
        const double old_val = sol[col];

        active = !((is_ge(coeff, 0) && is_eq(old_val, ub)) || (is_le(coeff, 0) && is_eq(lb, old_val)));
        if (active)
        {
            bool round_up = false;
            round_up = is_le(coeff, 0.0);

            assert(coeff != 0);
            // Exact value that makes slack zero
            fixval = old_val + slack_for_row / coeff;

            if (model.var_type[col] == 'I')
                fixval = static_cast<int>(fixval + (round_up ? 1 : 0));

            // Project to bounds
            fixval = fmin(ub, fmax(lb, fixval));

            /* score is valid only for threadIdx.x == 0 */
            score = compute_score_single_col_move(model, slack, sol, {fixval, col}, objective, sum_slack);
        }
    }
    /* Write violation to global memory. */
    if (thread_idx == 0 && active)
    {
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fixval, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* On exit, best_scores and best_random_moves contain for each block the best move and score found by the block. Consequently, best_scores and best_random_moves need to be larger than the grid dimension. */
__global__ void compute_random_moves_kernel(const GpuModelPtrs model, const double *slack, const double *sol, double objective, double sum_slack, solution_score *best_scores, single_col_move *best_random_moves, int n_cols, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState state;
    __shared__ single_col_move best_move;
    __shared__ solution_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        /* Set random seed to 0. */
        curand_init(0, 0, 0, &state);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    int n_cols_per_block = (n_cols + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_cols_start = min(n_cols, block_idx * n_cols_per_block);
    int my_cols_end = min(n_cols, (block_idx + 1) * n_cols_per_block);
    const int cols_range = my_cols_end - my_cols_start;

    /* Need at least one column! */
    if (cols_range > 0) {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
            const int col = my_cols_start + static_cast<int>((cols_range * curand_uniform(&state)));

            /* Compute a move for the picked column. */
            compute_random_move(model, state, slack, sol, objective, sum_slack, col, best_score, best_move, n_cols);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_random_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;
    }
}

/* On exit, best_scores and best oneopt move (greedy or feasible) contain for each block the best move and score found by the block. Consequently, best_scores and best_oneopt_moves need to be larger than the grid dimension.
TODO: specialize for n_moves >= n_cols */
template <const bool GREEDY>
__global__ void compute_oneopt_moves_kernel(const GpuModelPtrs model, const double *slack, const double *sol, double objective, double sum_slack, solution_score *best_scores, single_col_move *best_oneopt_moves, int n_cols, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState state;
    __shared__ single_col_move best_move;
    __shared__ solution_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        /* Set random seed to 0. */
        curand_init(0, 0, 0, &state);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    int n_cols_per_block = (n_cols + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_cols_start = min(n_cols, block_idx * n_cols_per_block);
    int my_cols_end = min(n_cols, (block_idx + 1) * n_cols_per_block);
    const int cols_range = my_cols_end - my_cols_start;

    /* Need at least one column! */
    if (cols_range > 0) {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
            const int col = my_cols_start + static_cast<int>((cols_range * curand_uniform(&state)));

            /* Compute a move for the picked column. */
            compute_oneopt_move<GREEDY>(model, state, slack, sol, objective, sum_slack, col, best_score, best_move, n_cols);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_oneopt_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;
    }
}


__global__ void compute_flip_moves_kernel(const GpuModelPtrs model, const double *slack, const double *sol,
                                                double objective, double sum_slack, solution_score *best_scores,
                                                single_col_move *best_flip_moves, int n_cols, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState state;
    __shared__ single_col_move best_move;
    __shared__ solution_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        /* Set random seed to 0. */
        curand_init(0, 0, 0, &state);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    int n_cols_per_block = (n_cols + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_cols_start = min(n_cols, block_idx * n_cols_per_block);
    int my_cols_end = min(n_cols, (block_idx + 1) * n_cols_per_block);
    const int cols_range = my_cols_end - my_cols_start;

    /* Need at least one column! */
    if (cols_range > 0) {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
            const int col = my_cols_start + static_cast<int>((cols_range * curand_uniform(&state)));

            /* Compute a move for the picked column. */
            compute_flip_move(model, slack, sol, objective, sum_slack, col, best_score, best_move, n_cols);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_flip_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;
    }
}

__global__ void compute_mtm_sat_moves_kernel(const GpuModelPtrs model, const double *slack, const double *sol,
                                                double objective, double sum_slack, solution_score *best_scores,
                                                single_col_move *best_mtm_moves, int n_cols, int n_rows, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState state;
    __shared__ single_col_move best_move;
    __shared__ solution_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        curand_init(0, 0, 0, &state);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    // int violated_count = end - valid_idx.begin();

    int n_rows_per_block = (n_rows + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_rows_start = min(n_cols, block_idx * n_rows_per_block);
    int my_rows_end = min(n_cols, (block_idx + 1) * n_rows_per_block);
    const int row_range = my_rows_end - my_rows_start;

    /* Need at least one row! */
    if (row_range > 0) {
        for (int move = 0; move < n_moves_per_block; ++move) {
            //TODO: make sure to pick a satisfied constraint
            /* Pick a row in our interval. This is uniformly distributed over [my_rows_start,...,my_rows_end). */
            const int row = my_rows_start + static_cast<int>(row_range * curand_uniform(&state));
            const int col_index = static_cast<int>(curand_uniform(&state) * (model.row_ptr[row + 1] - model.row_ptr[row]));
            /* Compute a move for the picked column. */
            compute_mtm_sat_move(model, slack, sol, objective, sum_slack, row, col_index, best_score, best_move, n_rows);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_mtm_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;
    }
}


__global__ void compute_mtm_unsat_moves_kernel(const GpuModelPtrs model, const double *slack, const double *sol,
                                               double objective, double sum_slack, const int *ind_violated_cons,
                                               const int size_violated_constraints, solution_score *best_scores,
                                               single_col_move *best_mtm_moves, int n_cols, int n_rows, int n_moves, curandState *states) {
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    // __shared__ curandState state;
    __shared__ single_col_move best_move;
    __shared__ solution_score best_score;
    __shared__ curandState random_state;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        random_state = states[block_idx];
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    // int violated_count = end - valid_idx.begin();

    // int n_rows_per_block = (n_rows + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    // TODO: this needs to be fixed!
    // int my_rows_start = min(n_cols, block_idx * n_rows_per_block);
    // int my_rows_end = min(n_cols, (block_idx + 1) * n_rows_per_block);
    const int row_range = size_violated_constraints;

    /* Need at least one row! */
    if (row_range > 0) {
        for (int move = 0; move < n_moves_per_block; ++move) {
            /* Pick a row in our interval. This is uniformly distributed over [my_rows_start,...,my_rows_end). */
            const int row = ind_violated_cons[curand(&random_state) % size_violated_constraints];
            const int col_index = curand(&random_state) % (model.row_ptr[row + 1] - model.row_ptr[row]);

            /* Compute a move for the picked column. */
            compute_mtm_unsat_move(model, slack, sol, objective, sum_slack, row, col_index, best_score, best_move, n_rows);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_mtm_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;

        /* Offload the changed random state for this block. */
        states[block_idx] = random_state;
    }
}


__global__ void apply_move(const GpuModelPtrs model, double *slack, double *sol, double objective, double sum_slack, solution_score *best_score, single_col_move *best_move, int n_cols)
{
    const int thread_idx = threadIdx.x;
    const double val = best_move->val;
    const int col = best_move->col;

    const double old_val = sol[col];
    assert(model.lb[col] <= val && val <= model.ub[col]);

    if (thread_idx == 0) {
        printf("Applying move jcol %d [%g, %g] : %g -> %g\n", col, model.lb[col], model.ub[col], old_val, val);
    }

    /* Iterate column and apply changes in slack. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* slack = rhs - Ax */
        slack[row_idx] += (old_val - val) * coef;
    }

    sol[col] = val;
}

double thrust_dot_product(const thrust::device_vector<double> &a,
                          const thrust::device_vector<double> &b)
{
    // Check sizes
    assert(a.size() == b.size());

    // Compute dot product: sum(a[i] * b[i])
    return thrust::inner_product(
        thrust::device,
        a.begin(), a.end(),
        b.begin(),
        0.0 /* Init value */
    );
}

/* Use cub to reduce violated constraints. We only check cub's buffersize once here. */
struct IsViolated {
    const double* slack;
    const char* row_sense;

    __device__ bool operator()(int idx) const {
        const int is_eq = row_sense[idx] == 'E';
        const double slack_row = slack[idx];

        /* If we have an equality, any slack counts. If we have an inequality, we only count negative slack rhs - Ax >= 0. Branching does not really matter here. */
        return is_eq ? !is_eq_feas(slack_row,0) : is_lt_feas(slack_row,0);  // TODO: use tolerances.h when merged from other branch.
    }
};

void EvolutionSearch::run()
{
    std::vector<double> sol_host(model_host.ncols, 0.0);

    for (int jcol = 0; jcol < model_host.ncols; ++jcol)
        sol_host[jcol] = max(model_host.lb[jcol], min(sol_host[jcol], model_host.ub[jcol]));

    std::vector<double> slacks_host = model_host.rhs;
    double sum_slack = 0.0;
    thrust::device_vector<int64_t> cub_n_selected(1);
    size_t cub_reduce_buffer_bytes = 0;

    /* Compute slacks_host = slacks_host - Ax = rhs - Ax */
    model_host.rows.SpMV(-1.0, sol_host.data(), slacks_host.data());

    int64_t n_violated_constraints = 0;
    for (int irow = 0; irow < model_host.nrows; ++irow)
    {
        const double slack_row = slacks_host[irow];
        FP_ASSERT(model_host.sense[irow] == 'L' || model_host.sense[irow] == 'E');

        if (model_host.sense[irow] == 'L' && is_lt_feas(slack_row,0)) {
            sum_slack += fabs(slack_row);
        }
        if (model_host.sense[irow] == 'E' && !is_eq_feas(slack_row,0)) {
            sum_slack += fabs(slack_row);
        }
    }
    consoleLog("Violated constraints: {}", n_violated_constraints);

    thrust::device_vector<double> sol_device = sol_host;
    auto gpu_model_ptrs = model_device.get_ptrs();

    consoleInfo("Starting evolution search on GPU");

    // TODO compute slacks, solution objective and violation
    double objective = thrust_dot_product(sol_device, model_device.objective);
    thrust::device_vector<double> slacks_device = slacks_host;

    thrust::device_vector<solution_score> best_scores_single_col(N_BLOCKS_SINGLE_COL_MOVE * AVAILABLE_MOVED, {DBL_MAX, DBL_MAX});
    thrust::device_vector<single_col_move> best_single_col_moves(N_BLOCKS_SINGLE_COL_MOVE * AVAILABLE_MOVED);
    thrust::device_vector<int> violated_constraints(model_device.nrows);

    cub::DeviceSelect::If(nullptr, cub_reduce_buffer_bytes, thrust::make_counting_iterator(0), thrust::raw_pointer_cast(violated_constraints.data()), thrust::raw_pointer_cast(cub_n_selected.data()), model_host.nrows, IsViolated{thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(gpu_model_ptrs.row_sense)});

    thrust::device_vector<std::uint8_t> cub_reduce_buffer(cub_reduce_buffer_bytes);

    /* Do some rounds:
     * get starting solutions (somehow zeros; lbs; ubs; lp_sol rounded; fpr solutions ...)
     *
     * while (true) {
     * Include solutions from CPU into pool?
     *      for some rounds:
     *        - GPU: sample moves (eval) over candidates (using local Search operators + others)
     *             -> kernel_local_search
     *             -> kernel_ALNS
     *        - GPU-reduce best moves and apply/CPU reduce best moves + apply
     * }
     *
     *
     */

    // TODO: sort ints and bins/extract them. Many moves only make sense for ints and bins!

    consoleLog("Initial slack     {}", sum_slack);
    consoleLog("Initial objective {}", objective);

    thrust::device_vector<curandState> random_states(N_MAX_MOVE_BLOCKS);
    constexpr int n_blocks_rng_init = (N_MAX_MOVE_BLOCKS + BLOCKSIZE_VECTOR_KERNEL - 1) / BLOCKSIZE_VECTOR_KERNEL;
    init_rng_per_block<<<n_blocks_rng_init, BLOCKSIZE_VECTOR_KERNEL>>>(thrust::raw_pointer_cast(random_states.data()), 1234UL);

    for (int round = 0; round < n_rounds; ++round) {
        int nmoves = 1e4;

        /* Gather violated rows. */
        cub::DeviceSelect::If(
            thrust::raw_pointer_cast(cub_reduce_buffer.data()), cub_reduce_buffer_bytes,
            thrust::make_counting_iterator(0), thrust::raw_pointer_cast(violated_constraints.data()),
            thrust::raw_pointer_cast(cub_n_selected.data()), model_host.nrows, IsViolated{
                thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(gpu_model_ptrs.row_sense)
            });

        /* This is a hidden copy ! */
        consoleLog("Found {} violated constraints", cub_n_selected[0]);

        /* ----- */
        // auto indi = violated_constraints;
        // for ( int index: indi) {
        //     consoleLog("{} ",  index);
        // }

        /* Compute best move for each block. */
        compute_random_moves_kernel<<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()),
            objective, sum_slack, thrust::raw_pointer_cast(best_scores_single_col.data()),
            thrust::raw_pointer_cast(best_single_col_moves.data()), model_host.ncols, nmoves);

        compute_oneopt_moves_kernel<true><<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()),
            objective, sum_slack, thrust::raw_pointer_cast(best_scores_single_col.data()) + N_BLOCKS_SINGLE_COL_MOVE,
            thrust::raw_pointer_cast(best_single_col_moves.data()) + N_BLOCKS_SINGLE_COL_MOVE, model_host.ncols,
            nmoves);

        compute_flip_moves_kernel<<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()),
            objective, sum_slack,
            thrust::raw_pointer_cast(best_scores_single_col.data()) + N_BLOCKS_SINGLE_COL_MOVE * 2,
            thrust::raw_pointer_cast(best_single_col_moves.data()) + N_BLOCKS_SINGLE_COL_MOVE * 2, model_host.ncols,
            nmoves);

        compute_mtm_unsat_moves_kernel<<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()),
            objective, sum_slack,
            thrust::raw_pointer_cast(violated_constraints.data()),
            cub_n_selected[0],
            thrust::raw_pointer_cast(best_scores_single_col.data()) + N_BLOCKS_SINGLE_COL_MOVE * 3,
            thrust::raw_pointer_cast(best_single_col_moves.data()) + N_BLOCKS_SINGLE_COL_MOVE * 3, model_host.ncols,
            model_host.nrows,
            nmoves, thrust::raw_pointer_cast(random_states.data()));

        compute_mtm_sat_moves_kernel<<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()),
            objective, sum_slack,
            thrust::raw_pointer_cast(best_scores_single_col.data()) + N_BLOCKS_SINGLE_COL_MOVE * 4,
            thrust::raw_pointer_cast(best_single_col_moves.data()) + N_BLOCKS_SINGLE_COL_MOVE * 4, model_host.ncols,
            model_host.nrows,
            nmoves);

        // // /* ----- */
        // thrust::host_vector<solution_score> host_scores = best_scores_single_col;
        // for ( auto &[objective, violation]: host_scores) {
        //     consoleLog("{} {}", objective, violation);
        // }


        /* Reduce best moves to get globally best move. */
        auto max_iter = thrust::min_element(thrust::device, best_scores_single_col.begin(),
                                            best_scores_single_col.end(),
                                            [] __device__(const solution_score &a, const solution_score &b) {
                                                return a.feas_score() < b.feas_score();
                                            });

        int min_index = max_iter - best_scores_single_col.begin();
        solution_score score = (*max_iter); // Hidden copy GPU -> CPU
        double min_value = score.feas_score();

        std::string move_name;
        if (min_index >= 4 * N_BLOCKS_SINGLE_COL_MOVE)
            move_name = "mtm_sat";
        else if (min_index >= 3 * N_BLOCKS_SINGLE_COL_MOVE)
            move_name = "mtm_unsat";
        else if (min_index >= 2 * N_BLOCKS_SINGLE_COL_MOVE)
            move_name = "flip";
        else if (min_index >= 1 * N_BLOCKS_SINGLE_COL_MOVE)
            move_name = "oneopt";
        else if (min_index >= 0)
            move_name = "random";
        else
            move_name = "unknown";

        consoleLog("Taking {} move", move_name);
        consoleLog("(idx, score): {} {}", min_index, min_value);

        /* Apply best move. */
        apply_move<<<1, 1024>>>(gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()),
                                thrust::raw_pointer_cast(sol_device.data()), objective, sum_slack,
                                thrust::raw_pointer_cast(best_scores_single_col.data()) + min_index,
                                thrust::raw_pointer_cast(best_single_col_moves.data()) + min_index, model_host.ncols);

        objective = score.objective;
        sum_slack = score.violation;

        consoleLog("(objective, sum_slack): {} {}", objective, sum_slack);
    }
};
