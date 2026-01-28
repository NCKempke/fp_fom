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

constexpr int N_BLOCKS_SINGLE_COL_MOVE = 512;
constexpr int BLOCKSIZE_SINGLE_COL_MOVE = 32;

/* TODO:
 * - oneopt (feas); implemented but not applied
 * - mixed tight move: one cons + one var -> make cons as feasible as possible
 * - swap move?
 * -
 * - TSP swap?
 * - Avoid duplicate moves.
 *
 * - Solution pool;
 * - Sync solutions from FPR;
 * - Scoring function;
 */

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
    //     // printf("fixval: %g; colval: %g\n", fixval, col_val);
    // }

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
    if ((obj > 0.0 && col_val == lb) || (obj < 0.0 && col_val == ub) || obj == 0.0)
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

__global__ void apply_move(const GpuModelPtrs model, double *slack, double *sol, double objective, double sum_slack, solution_score *best_score, single_col_move *best_move, int n_cols)
{
    const int thread_idx = threadIdx.x;
    const double val = best_move->val;
    const int col = best_move->col;

    const double old_val = sol[col];

    assert(model.lb[col] <= val && val <= model.ub[col]);

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

void EvolutionSearch::run()
{
    std::vector<double> sol_host(model_host.ncols, 0.0);

    for (int jcol = 0; jcol < model_host.ncols; ++jcol)
        sol_host[jcol] = max(model_host.lb[jcol], min(sol_host[jcol], model_host.ub[jcol]));

    std::vector<double> slacks_host = model_host.rhs;
    double sum_slack = 0.0;

    /* Compute slacks_host = slacks_host - Ax = rhs - Ax */
    model_host.rows.SpMV(-1.0, sol_host.data(), slacks_host.data());

    for (int irow = 0; irow < model_host.nrows; ++irow)
    {
        const double slack_row = slacks_host[irow];
        FP_ASSERT(model_host.sense[irow] == 'L' || model_host.sense[irow] == 'E');

        if (model_host.sense[irow] == 'E' || slack_row < 0.0)
            sum_slack += fabs(slack_row);
    }

    thrust::device_vector<double> sol_device = sol_host;
    auto gpu_model_ptrs = model_device.get_ptrs();

    consoleInfo("Starting evolution search on GPU");

    // TODO compute slacks, solution objective and violation
    double objective = thrust_dot_product(sol_device, model_device.objective);
    thrust::device_vector<double> slacks_device = slacks_host;

    thrust::device_vector<solution_score> best_scores_single_col(N_BLOCKS_SINGLE_COL_MOVE * 2, {DBL_MAX, DBL_MAX});
    thrust::device_vector<single_col_move> best_single_col_moves(N_BLOCKS_SINGLE_COL_MOVE * 2);

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

    for (int round = 0; round < n_rounds; ++round)
    {
        /* Compute best move for each block. */
        compute_random_moves_kernel<<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()), objective, sum_slack, thrust::raw_pointer_cast(best_scores_single_col.data()), thrust::raw_pointer_cast(best_single_col_moves.data()), model_host.ncols, 1e6);

        compute_oneopt_moves_kernel<true><<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()), objective, sum_slack, thrust::raw_pointer_cast(best_scores_single_col.data()) + N_BLOCKS_SINGLE_COL_MOVE, thrust::raw_pointer_cast(best_single_col_moves.data()) + N_BLOCKS_SINGLE_COL_MOVE, model_host.ncols, 1e6);

        /* ----- */

        // thrust::host_vector<solution_score> host_scores = best_scores_single_col;

        // for (auto &sol : host_scores)
        // {
        //     consoleLog("{} {}", sol.objective, sol.violation);
        // }

        /* Reduce best moves to get globally best move. */
        auto max_iter = thrust::min_element(thrust::device, best_scores_single_col.begin(), best_scores_single_col.end(), [] __device__(const solution_score &a, const solution_score &b)
                                            { return a.feas_score() < b.feas_score(); });

        int min_index = max_iter - best_scores_single_col.begin();
        solution_score score = (*max_iter); // Hidden copy GPU -> CPU
        double min_value = score.feas_score();
        consoleLog("(idx, score): {} {}", min_index, min_value);

        /* Apply best move. */
        apply_move<<<1, 1024>>>(gpu_model_ptrs, thrust::raw_pointer_cast(slacks_device.data()), thrust::raw_pointer_cast(sol_device.data()), objective, sum_slack, thrust::raw_pointer_cast(best_scores_single_col.data()) + min_index, thrust::raw_pointer_cast(best_single_col_moves.data()) + min_index, model_host.ncols);

        objective = score.objective;
        sum_slack = score.violation;

        consoleLog("(objective, sum_slack): {} {}", objective, sum_slack);
    }
};