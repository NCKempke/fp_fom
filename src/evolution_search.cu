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
#include <cub/util_device.cuh>        // CountingInputIterator
#include <cub/device/device_select.cuh> // DeviceSelect::Flagged

constexpr int N_BLOCKS_SINGLE_COL_MOVE = 512;
constexpr int BLOCKSIZE_SINGLE_COL_MOVE = 32;
constexpr int AVAILABLE_MOVED = 4;

/* Moves:
 * - one_opt (feas)   : implemented not applied
 * - one_opt (greedy) : push variable in direction of its objective
 * - flip             : flips a binary randomly selected variable
 * - random           : selects a random variable and assigns it a random value
 * - mtm_unsatisfied  : selects a random violated constraint, then selects a variable within its range
 *                      and adjusts it to make the constraint as feasible as possible TODO: testing it since prob of hitting
 *
 * TODO:
 * - mtm_satisfied    :
 * - swap             : select two (binary) variables with different values and swap them
 *
 * - TSP swap?
 * - Avoid duplicate moves.
 *
 * - Solution pool;
 * - Sync solutions from FPR;
 * - Scoring function;
 */

constexpr double FEASTOL = 1e-6;
constexpr double EPSILON = 1e-9;

struct single_col_move
{
    double val;
    int col;
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


// Device functions for comparisons with absolute tolerance only
__device__ bool isEq(double a, double b) {
    return fabsf(a - b) <= EPSILON;
}

__device__ bool isGE(double a, double b) {
    // a >= b considering tolerance
    return a-b >= -EPSILON;
}

__device__ bool isGT(double a, double b) {
    // a >= b considering tolerance
    return a-b > -EPSILON;
}

__device__ bool isLE(double a, double b) {
    // a <= b considering tolerance
    return a-b <= EPSILON;
}

__device__ bool isLT(double a, double b) {
    // a <= b considering tolerance
    return a-b < EPSILON;
}
// Device functions for comparisons with absolute tolerance only
__device__ bool isFeasEq(double a, double b) {
    return fabsf(a - b) <= FEASTOL;
}

__device__ bool isFeasGE(double a, double b) {
    // a >= b considering tolerance
    return a-b >= -FEASTOL;
}

__device__ bool isFeasLE(double a, double b) {
    // a <= b considering tolerance
    return a-b <= FEASTOL;
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

    // if (thread_idx == 0)
    // {
    //     printf("fixval: %g; colval: %g\n", move.val, col_val);
    // }

    //TODO: is this correct to skip here?
    if (isEq(delta, 0))
        return{objective, slack_change_thread};
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

    // TODO: fixval != col_val
    const double fixval = static_cast<int>((lb + (ub - lb) * random_val) + 0.5);

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
    if ((isGT(obj, 0.0) && isEq(col_val, lb)) || (isLT(obj, 0.0) && isEq(col_val, ub)) || isEq(obj, 0.0))
        return;

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

    // TODO column must be integer
    double lb = model.lb[col];
    double ub = model.ub[col];

    // is binary variable?
    if (model.var_type[col] == 'I' && lb == 0 && ub == 1)
        return;
    const double fixval = 1 - sol[col];

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


__device__ void compute_mtm_unsat_move(const GpuModelPtrs &model, const double *slack, const double *sol, const double objective, const double sum_slack, const int row, const int col_index, solution_score
                                  &best_score, single_col_move &best_move, const int nrows)
{
    const int thread_idx = threadIdx.x;

    if (row >= nrows)
        return;

    const double slack_for_row = slack[row];
    // skip satisfied constraints -> either equation with slack == 0 or inequalities with positive slack
    if ((model.row_sense[row] == 'E' &&isFeasEq(slack_for_row, 0))  || isGT(slack_for_row, 0) )
        return;

    // get variable
    const int col = model.col_idx[model.row_ptr[row] + col_index];
    double coeff = model.row_val[model.row_ptr[row] + col_index];

    const double lb = model.lb[col];
    const double ub = model.ub[col];
    const double old_val = sol[col];

    bool round_up = false;
    // If a >= 0, decreasing x increases slack in a x <= b; skip if x is already at its lower bound. (analogous for a <= 0)
    // if (model.row_sense[row] != 'E' or (model.row_sense[row] != 'E' &&isGT(slack_for_row, 0))) {
    if (!(model.row_sense[row] == 'E' &&isGE(slack_for_row, 0))) {
        if ((isLE(coeff, 0) &&isEq(old_val, ub)) || (isGE(coeff , 0) &&isEq(lb, old_val)))
            return;
        round_up = isLE(coeff, 0.0);
    }
    else {
        assert(slack_for_row > 0.0);
            if ( (isGE(coeff, 0) &&isEq(old_val, ub)) || (isLE(coeff , 0) &&isEq(lb, old_val)) )
                return;
        round_up = isGE(coeff, 0.0);
    }

    // Exact value that makes slack zero
    double fixval = old_val - slack_for_row / coeff;

    if (model.var_type[col] == 'I')
        fixval = static_cast<int>(fixval + (round_up ? 1 : 0));

    // printf("col %d in row  %d fixed to  %f", row, col, fixval);
    // Project to bounds
    fixval = fmin(ub, fmax(lb, fixval));

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

    if (my_cols_start >= my_cols_end)
        return;

    for (int move = 0; move < n_moves_per_block; ++move)
    {
        /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
        const int col = my_cols_start + static_cast<int>((cols_range * curand_uniform(&state)));

        /* Compute a move for the picked column. */
        compute_random_move(model, state, slack, sol, objective, sum_slack, col, best_score, best_move, n_cols);
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

    if (my_cols_start >= my_cols_end)
        return;

    for (int move = 0; move < n_moves_per_block; ++move)
    {
        /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
        const int col = my_cols_start + static_cast<int>((cols_range * curand_uniform(&state)));

        /* Compute a move for the picked column. */
        compute_oneopt_move<GREEDY>(model, state, slack, sol, objective, sum_slack, col, best_score, best_move, n_cols);
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

    if (my_cols_start >= my_cols_end)
        return;

    for (int move = 0; move < n_moves_per_block; ++move)
    {
        /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
        const int col = my_cols_start + static_cast<int>((cols_range * curand_uniform(&state)));

        /* Compute a move for the picked column. */
        compute_flip_move(model, slack, sol, objective, sum_slack, col, best_score, best_move, n_cols);
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_flip_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;
    }
}

__global__ void compute_mtm_unsat_moves_kernel(const GpuModelPtrs model, const double *slack, const double *sol,
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

    if (my_rows_start >= my_rows_end)
        return;

    for (int move = 0; move < n_moves_per_block; ++move) {
        /* Pick a row in our interval. This is uniformly distributed over [my_rows_start,...,my_rows_end). */
        const int row = my_rows_start + static_cast<int>(row_range * curand_uniform(&state));
        // TODO: Fix incorrect constraint distribution. As an optimization, cache violated constraints for quicker retrieval?
        // generate a column within the range (not necessarily helpful)
        const int col_index = static_cast<int>(curand_uniform(&state) * (model.row_ptr[row + 1] - model.row_ptr[row]));

        // if (thread_idx == 0)
        //     printf("col %d row %d move %d  %d\n", col_index, row, move, n_moves_per_block);

        /* Compute a move for the picked column. */
        compute_mtm_unsat_move(model, slack, sol, objective, sum_slack, row, col_index, best_score, best_move, n_rows);
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_mtm_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;
    }
}


__global__ void apply_move(const GpuModelPtrs model, double *slack, double *sol, double objective, double sum_slack, solution_score *best_score, single_col_move *best_move, int n_cols)
{
    const int thread_idx = threadIdx.x;
    const double val = best_move->val;
    const int col = best_move->col;

    const double old_val = sol[col];

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
        return (is_eq ? fabs(slack_row) : -slack_row) > 1e-6;  // TODO: use tolerances.h when merged from other branch.
    }
};

void EvolutionSearch::run()
{
    std::vector<double> sol_host(model_host.ncols, 0.0);
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

        if (model_host.sense[irow] == 'L' && -slack_row > 1e-6) {
            sum_slack += fabs(slack_row);
            n_violated_constraints++;
        }
        if (model_host.sense[irow] == 'E' && fabs(slack_row) > 1e-6) {
            sum_slack += fabs(slack_row);
            n_violated_constraints++;
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

    consoleLog("Initial slack     {}", sum_slack);
    consoleLog("Initial objective {}", objective);

    for (int round = 0; round < n_rounds; ++round) {
        int nmoves = 1e3;

        /* Gather violated rows. */
        cub::DeviceSelect::If(
            thrust::raw_pointer_cast(cub_reduce_buffer.data()), cub_reduce_buffer_bytes, thrust::make_counting_iterator(0), thrust::raw_pointer_cast(violated_constraints.data()), thrust::raw_pointer_cast(cub_n_selected.data()), model_host.nrows, IsViolated{thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(gpu_model_ptrs.row_sense)});

        /* This is a hidden copy ! */
        consoleLog("Found {} violated constraints", cub_n_selected[0]);

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
            thrust::raw_pointer_cast(best_scores_single_col.data()) + N_BLOCKS_SINGLE_COL_MOVE * 3,
            thrust::raw_pointer_cast(best_single_col_moves.data()) + N_BLOCKS_SINGLE_COL_MOVE * 3, model_host.ncols,
            model_host.nrows,
            nmoves);

        /* ----- */
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
