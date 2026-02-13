#include "evolution_search.cuh"

#include "gpu_data.cuh"
#include "mip.h"
#include "move_type.h"
#include "utils.cuh"

#include <consolelog.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/random.h>

#include "timer.h"
#include "cub/cub.cuh"

extern int UserBreak;
#define EXTENDED_DEBUG

/* We submit blocks with WARPS_PER_BLOCK many warps. Each warp is responsible for computing N_MOVES_PER_WARP many moves.
* To compute n_moves of a certain move type, we need to submit
*
*    (n_moves + N_MOVES_PER_SINGLE_COL_BLOCK - 1) / N_MOVES_PER_SINGLE_COL_BLOCK
*
* blocks.
*/
constexpr int BLOCKSIZE_MOVE = 256;


constexpr int N_WARPS_PER_BLOCK = BLOCKSIZE_MOVE / WARP_SIZE;
static_assert(BLOCKSIZE_MOVE % WARP_SIZE == 0);

constexpr int N_MOVES_PER_WARP = 32;
constexpr int N_MOVES_PER_SINGLE_COL_BLOCK = N_MOVES_PER_WARP * N_WARPS_PER_BLOCK;

/* CUDA graph setup. */
constexpr bool GRAPH_ENABLE = true; /* Turn off for debugging and extra output. */
constexpr int GRAPH_N_ITER = 100; /* Amount of iterations done per graph submission. */

/* Evolution search specific parameters. */
constexpr int LP_SOLUTION_FREQ = 1000;
constexpr int SOLUTION_TRANSFER_FREQ = 10;
constexpr int SOLUTION_IMPORT_FREQ = 100;
constexpr int MAX_VALUE_HUGE = 1000;
constexpr int RECOMPUTE_SOL_METRICS_FREQ = SOLUTION_TRANSFER_FREQ / 10;
static_assert(SOLUTION_TRANSFER_FREQ % RECOMPUTE_SOL_METRICS_FREQ == 0);

constexpr int REMOVE_CROSSOVER_FREQ = 100 ;

constexpr double SMOOTHING_PROBABILITY = 0.0001;

struct warp_sampling_range {
    int beg;
    int end;
};

/* For a given interval of sampling candidates [1,..,n_candidates) (e.g. rows or columns) and n_samples total to be computed samples,
 * determine for each warp in this block its assigned sampling range and its assigned number of samples. Returns {beg, end, n_samples}. */
__device__ inline warp_sampling_range get_warp_sampling_range(const int n_candidates, const int n_samples) {
    warp_sampling_range range{0, 0};

    const int block_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    const int warp_id_block = thread_idx / WARP_SIZE;
    const int warp_id_global = block_idx * N_WARPS_PER_BLOCK + warp_id_block;

    /* Only the first n_active_warps actually compute samples. We compute at least n_samples moves but one warp always computes its full N_MOVES_PER_WARP. */
    const int n_active_warps = (n_samples + N_MOVES_PER_WARP - 1) / N_MOVES_PER_WARP;

    if (warp_id_global >= n_active_warps) {
        return range;
    }

    /* Partition [1,..,n_candidates) among all active warps. */
    const int base = n_candidates / n_active_warps;
    const int remaining = n_candidates % n_active_warps;

    if (warp_id_global < remaining) {
        range.beg = warp_id_global * (base + 1);
        range.end = range.beg + (base + 1);
    } else {
        range.beg = remaining * (base + 1) + (warp_id_global - remaining) * base;
        range.end = range.beg + base;
    }

    range.beg = min(range.beg, n_candidates);
    range.end = min(range.end, n_candidates);

    assert(range.beg <= range.end);
    return range;
}

/* For a given range [beg,..,end) return N_MOVES_PER_WARP draws (unique) starting at draws[warp_id * WARP_SIZE]*/
__device__ void warp_sample_range(int* draws, int beg, int end, size_t seed)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    int* draws_warp = &draws[warp_id * N_MOVES_PER_WARP]; /* pointer to this warp's draws */
    const int range = end - beg;

    if (range <= N_MOVES_PER_WARP) {
        /* Case 1: small range, just return all entries from the range */
        for (int i = thread_idx_warp; i < N_MOVES_PER_WARP; i += WARP_SIZE) {
            draws_warp[i] = i < range ? beg + i : -1;
        }
    }
    else {
        curandState state_thread;
        const int global_thread_id = blockIdx.x * blockDim.x + thread_idx;
        curand_init(seed, global_thread_id, 0, &state_thread);

        const int chunk_size = range / N_MOVES_PER_WARP;
        const int leftover = range % N_MOVES_PER_WARP;

        /* Case 2: larger range, draw N_MOVES_PER_WARP samples in parallel and uniquely.
         * For this, we split the range into N_MOVES_PER_WARP non-overlapping intervals and each thread picks one column from its sub-range. This enforces some additional uniformness for the draw but well. */
        for (int i = thread_idx_warp; i < N_MOVES_PER_WARP; i += WARP_SIZE) {
            /* Compute this thread's sub-range [thread_beg, thread_end) for this sample. */
            const int thread_chunk_size = chunk_size + (i < leftover ? 1 : 0);

            // Compute start of this thread's chunk
            const int thread_beg = i * chunk_size + min(i, leftover);
            const int thread_end = thread_beg + thread_chunk_size;
            assert(thread_beg < thread_end);

            draws_warp[i] = beg + thread_beg + get_random_int_thread(state_thread, thread_end - thread_beg);
            assert(beg <= draws_warp[i] && draws_warp[i] < end);
        }
    }

    __syncwarp(); /* Ensure warp is done and synchronize its changes. */
}

/* Moves:
 * - one_opt (feas)   : push variable in direction of its objective while maintaining feasibility
 * - one_opt (greedy) : push variable in direction of its objective
 * - flip             : flips a binary randomly selected variable
 * - random           : selects a random variable and assigns it a random value
 * - mtm_satisfied    : select a random satisfied constraint and set slack to zero; iterates all variables
 *                      of the constraint to find the best candidate
 * - mtm_unsatisfied  : selects a random violated constraint, then selects a variable within its range
 *                      to make the constraint as feasible as possible; iterates all variables of the
 *                      constraint to find the best candidate
 *
 * TODO:
 * - swap             : select two (binary) variables with different values and swap them
 * - Lagrange         : from Feaspump
 * - TSP swap?        : swaps 4 binaries I think.
 *
 * - Avoid duplicate moves:
 *    -- random moves only for integers and continuous columns
 *    -- one_opt (either) only for integers and continuous columns
 *    -- flip only for binary columns
 *    -- mtm_satisfied only for integers and continuous within the constraint
 *    -- mtm_unsatisfied only for integers and continuous within the constraint
 *
 * - Solution pool;
 * TODO:
 * - manage solutions (decide which solutions to keep and which to discard):
 *    -- Sync solutions from/to global solution pool;
 *    -- export solution back to FPR
 *    -- Mutate solutions in pool after n rounds (crossover)
 *
 * - Scoring function:
 *    -- secondary score from Local-MIP?
 *
 * Implementation improvements
 *    -- use cuda graph to submit 100-ish rounds at once
 */
int blocks_for_samples(const int n_samples_for_type) {
    if (n_samples_for_type <= 0)
        return 0;

    return (n_samples_for_type + N_MOVES_PER_SINGLE_COL_BLOCK - 1) / N_MOVES_PER_SINGLE_COL_BLOCK;
}

/* Return whether a column was marked tabu. */
__device__ inline bool is_tabu(const int *tabu_col, const int col, const int iter, const int tabu_tenure)
{
    return tabu_col[col] > iter - tabu_tenure;
}

__device__ void reduce_and_offload_best_score_in_block(move_score* best_score, single_col_move* best_move, move_config& config) {

    /* Reduce best_move and best_score among the whole block. */
    __syncthreads();

    /* offload the best move and its score to main memory; done by the block's thread 0 */
    if (threadIdx.x == 0)
    {
        move_score block_best_score = best_score[0];
        single_col_move block_best_move = best_move[0];

        for (int warp = 1; warp < N_WARPS_PER_BLOCK; ++warp)
        {
            if (best_score[warp].is_lt_feas_score(block_best_score))
            {
                block_best_score = best_score[warp];
                block_best_move  = best_move[warp];
            }
        }

        /* offload the best move and its score to main memory */
        config.best_score[blockIdx.x] = block_best_score;
        config.best_move[blockIdx.x] = block_best_move;
    }
}

TabuSearchDataDevice::TabuSearchDataDevice(const int nrows_, const int ncols_, const int tabu_tenure)
    : best_sol(ncols_, 0.0),
      current_sol(ncols_, 0.0),
      slacks(nrows_, 0.0),
      tabu(ncols_, -tabu_tenure),
      constraint_weights(nrows_, 1),
      violated_constraints(nrows_)
{
    for (size_t i = 0; i < AVAILABLE_MOVES; ++i)
        cudaStreamCreate(&streams[i]);

    thrust::sequence(thrust::cuda::par.on(streams.front()), violated_constraints.begin(), violated_constraints.end());
};

TabuSearchDataDevice::~TabuSearchDataDevice()
{
    for (size_t i = 0; i < AVAILABLE_MOVES; ++i)
        cudaStreamDestroy(streams[i]);
};

/* Returns TabuSearchKernelArgs which lives on device! Needs to be freed after use. */
TabuSearchKernelArgs *create_args_and_copy_to_device(TabuSearchDataDevice &data, const MIPInstance &mip, const int tabu_tenure)
{
    TabuSearchKernelArgs args{};

    args.best_sol = thrust::raw_pointer_cast(data.best_sol.data());
    args.best_objective = INFTY;
    args.best_violation = INFTY;
    args.is_found_feasible = 0;

    args.current_sol = thrust::raw_pointer_cast(data.current_sol.data());
    args.slacks = thrust::raw_pointer_cast(data.slacks.data());
    args.tabu = thrust::raw_pointer_cast(data.tabu.data());

    args.constraint_weights = thrust::raw_pointer_cast(data.constraint_weights.data());
    args.objective_weight = 1.0;

    args.violated_constraints = thrust::raw_pointer_cast(data.violated_constraints.data());

    args.sum_viol = 0.0;
    args.objective = 0.0;

    args.n_violated = 0;
    args.iter = 1;

    args.nrows = mip.nrows;
    args.n_equalities = mip.n_equalities;

    args.ncols = mip.ncols;
    args.n_binaries = mip.n_binaries;
    args.n_integers = mip.n_integers;

    args.tabu_tenure = tabu_tenure;

    TabuSearchKernelArgs *device_args;
    cudaMalloc(&device_args, sizeof(TabuSearchKernelArgs));
    CHECK_CUDA(cudaMemcpyAuto(device_args, &args));

    return device_args;
}

/* Returns for threadIdx.x % WARP_SIZE == 0 the score after virtually applying give move. Runs on a per-warp basis and expects equal arguments across the warp. */
__device__ move_score compute_score_single_col_move_warp(const GpuModelPtrs &model, const TabuSearchKernelArgs* args, const single_col_move move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    const int col = move.col;

    const double obj_coef = model.objective[col];
    const double col_val = args->current_sol[col];
    const double delta = move.val - col_val;
    const double delta_obj = delta * obj_coef;
    double viol_change_thread = 0.0;
    double weighted_viol_change_thread = 0.0;

    assert(model.lb[col] <= move.val && move.val <= model.ub[col]);

    if (is_eq(delta, 0))
        return {0.0, 0.0, 0.0};

    /* Iterate column and compute changes in violation. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx_warp; inz < col_end; inz += WARP_SIZE)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];
        const double weight = args->constraint_weights[row_idx];

        /* We have <= and = only. */
        const double is_eq = row_idx < args->n_equalities;

        const double slack_old = args->slacks[row_idx];
        const double slack_new = slack_old - coef * delta;

        const double viol_old = is_eq * fabs(slack_old) + (1 - is_eq) * fmax(0.0, -slack_old);
        const double viol_new = is_eq * fabs(slack_new) + (1 - is_eq) * fmax(0.0, -slack_new);

        viol_change_thread += (viol_new - viol_old);

        const bool feas_before = is_zero_feas(viol_old);
        const bool feas_after = is_zero_feas(viol_new);

        /* If the constraint became feasible, we subtract 1 * weight or 2 * weight, if it became infeasible, we add 1 * weight or 2 * weight.
         * If it became more feasible, we add 1/2 * weight or 1 * weight. */
        if (feas_before && !feas_after) {
            /* Bad move. Increases score. */
            weighted_viol_change_thread += (1.0 + is_eq) * weight;

        } else if (!feas_before && feas_after) {
            weighted_viol_change_thread -= (1.0 + is_eq) * weight;
        } else if (!feas_before && !feas_after && viol_new < viol_old) {
            weighted_viol_change_thread -= (0.5 + 0.5 * is_eq) * weight;
        }
    }

    const double slack_change = warp_sum_reduce(viol_change_thread);
    const double weighted_viol_change = warp_sum_reduce(weighted_viol_change_thread);

    return {delta_obj, slack_change, weighted_viol_change};
}

/* Returns for threadIdx.x == 0 the score after virtually applying give move. */
__device__ move_score compute_score_col_swap(const GpuModelPtrs &model, const TabuSearchKernelArgs *args, const double *slack, const double *sol, const swap_move move)
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
    //     printf("fix_val: %g; colval: %g\n", move.val, col_val);
    // }

    // TODO: is this correct to skip here?
    if (is_eq(delta1, 0))
    {
        assert(delta2 == 0);
        return {0.0, 0.0, 0.0};
    }
    /* Iterate column and compute changes in violation. */

    // TODO: write function here to reduce duplicates
    const int col_beg = model.row_ptr_trans[col1];
    const int col_end = model.row_ptr_trans[col1 + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* We have <= and = only. */
        const int is_eq = row_idx < args->n_equalities;

        const double slack_old = slack[row_idx];
        const double slack_new = slack_old - coef * delta1;

        const double viol_old = is_eq * abs(slack_old) + (1 - is_eq) * max(0.0, -slack_old);
        const double viol_new = is_eq * abs(slack_new) + (1 - is_eq) * max(0.0, -slack_new);

        slack_change_thread += (viol_new - viol_old);
    }

    const int col_beg_2 = model.row_ptr_trans[col2];
    const int col_end_2 = model.row_ptr_trans[col2 + 1];

    for (int inz = col_beg_2 + thread_idx; inz < col_end_2; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* We have <= and = only. */
        const int is_eq = row_idx < args->n_equalities;

        const double slack_old = slack[row_idx];
        const double slack_new = slack_old - coef * delta2;

        double viol_old = is_eq * abs(slack_old) + (1 - is_eq) * max(0.0, -slack_old);
        double viol_new = is_eq * abs(slack_new) + (1 - is_eq) * max(0.0, -slack_new);

        slack_change_thread += (viol_new - viol_old);
    }

    using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_MOVE>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    const double slack_change = BlockReduce(temp_storage).Sum(slack_change_thread);

    // TODO weighted score
    return {delta_obj, slack_change, 0.0};
}

/* activities = rhs - Ax */
__device__ void compute_random_move(const GpuModelPtrs &model, curandState &random_state, const TabuSearchKernelArgs *args, const int col, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    assert(col < args->ncols);

    double lb = model.lb[col];
    double ub = model.ub[col];

    if (lb < 1000 || 1000 < ub)
        return;

    /* Don't run if these are too close. */
    if (ub - lb < 0.001)
        return;

    double fix_val;
    double col_val = args->current_sol[col];

    if (col < args->n_binaries)
    {
        fix_val = 1.0 - col_val;
    }
    else if (col < args->n_binaries + args->n_integers)
    {
        const int ilb = static_cast<int>(lb);
        const int iub = static_cast<int>(ub);
        const int ival = static_cast<int>(col_val);

        const int r = get_random_int_warp(random_state, iub - ilb);
        int fix_i = ilb + r;
        if (fix_i >= ival)
            fix_i++;

        fix_val = static_cast<double>(fix_i);
    }
    else
    {
        fix_val = get_random_double_in_range_warp(random_state, lb, ub);
    }

    assert(lb <= fix_val && fix_val <= ub);

    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move_warp(model, args, {fix_val, col});

    /* Write violation to smem. */
    if (thread_idx_warp == 0)
    {
        // printf("Compute score %g %g %g", score.objective_change, score.violation_change, score.weighted_violation_change);
        if (score.is_lt_feas_score(best_score))
        {
            best_score = score;
            best_move = {fix_val, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx_warp == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* activities = rhs - Ax */
template <const bool GREEDY>
__device__ void compute_oneopt_move(const GpuModelPtrs &model, const TabuSearchKernelArgs *args, const int col, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    assert(col < args->ncols);

    // TODO column must be integer ?
    const double col_val = args->current_sol[col];
    const double lb = model.lb[col];
    const double ub = model.ub[col];
    const double obj = model.objective[col];

    /* Only do this for variables with non-zero objective. */
    if ((is_gt(obj, 0.0) && is_eq(col_val, lb)) || (is_lt(obj, 0.0) && is_eq(col_val, ub)) || is_eq(obj, 0.0))
        return;

    assert(lb <= col_val && col_val <= ub);

    /* Get minimum slack in locking direction. Iterate column i and for each row j that locks in objective direction and compute min_j(row_coeff_ij * slack_j). */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    if (lb == -INFTY || ub == INFTY)
        return;

    /* Initialize stepsize with its greedy value. */
    double stepsize = obj > 0.0 ? col_val - lb : ub - col_val;

    if (!GREEDY)
    {
        for (int inz = col_beg + thread_idx_warp; inz < col_end; inz += WARP_SIZE)
        {
            const double coef = model.row_val_trans[inz];
            const int row_idx = model.col_idx_trans[inz];
            const double row_slack = args->slacks[row_idx];
            const int is_eq = row_idx < args->n_equalities;
            const int is_objcoef_pos = (obj * coef > 0.0);
            const int is_row_slack_neg = is_lt_feas(row_slack, 0.0);

            /* set stepsize to 0 if any row is an equality or infeasible. */
            const int skip = (is_eq | is_row_slack_neg);
            const double scaled_slack = fabs(coef * row_slack);

            // TODO : maybe allow infeasible solutions here and allow becoming infeasible
            // TODO: rounding is not considered yet
            stepsize = min(stepsize, (1 - skip) * (is_objcoef_pos * stepsize + (1 - is_objcoef_pos) * scaled_slack));
        }

        /* Reduce min of stepsize; this is done per warp for now. */
        stepsize = warp_min_reduce(stepsize);
    }

    assert(is_ge(stepsize, 0.0));
    if (is_zero(stepsize))
        return;

    const double fix_val = obj > 0.0 ? col_val - stepsize : col_val + stepsize;

    assert(is_le(lb, fix_val) && is_le(fix_val, ub));

    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move_warp(model, args, {fix_val, col});

    /* Write violation to smem. */
    if (thread_idx_warp == 0)
    {
        if (score.is_lt_feas_score(best_score))
        {
            best_score = score;
            best_move = {fix_val, col};
        }
        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* TODO: delete flip move? This is completely equal to the binary random move. */
__device__ void compute_flip_move(const GpuModelPtrs &model, const TabuSearchKernelArgs* args, const int col, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;

    /* Only for binaries. */
    if (col >= args->n_binaries)
        return;

    const double fix_val = args->current_sol[col] > 0.5 ? 0 : 1;

    /* score is valid only for threadIdx.x == 0 */
    const move_score score = compute_score_single_col_move_warp(model, args, {fix_val, col});

    /* Write violation to smem. */
    if (thread_idx_warp == 0)
    {
        if (score.is_lt_feas_score(best_score))
        {
            best_score = score;
            best_move = {fix_val, col};
        }
        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* Compute all possible mtm moves for a given constraint. */
__device__ void compute_mtm_move(const GpuModelPtrs &model, const TabuSearchKernelArgs* args, const int row, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;

    assert(row < args->nrows);

    const double slack_for_row = args->slacks[row];
    const bool slack_is_pos = is_gt_feas(slack_for_row, 0);

    if (is_zero(slack_for_row))
        return;

    for (int inz = model.row_ptr[row]; inz < model.row_ptr[row + 1]; ++inz) {
        const int col = model.col_idx[inz];

        if (is_tabu(args->tabu, col, args->iter, args->tabu_tenure))
            continue;

        const double coeff = model.row_val[inz];
        const double lb = model.lb[col];
        const double ub = model.ub[col];
        const double old_val = args->current_sol[col];

        move_score score;
        bool move_up;

        /* Try to move col as far as possible to make the constraint exactly tight/feasible; as we know the row is infeasible, move the slack rhs - a'x towards zero. */
        if (slack_is_pos)
        {
            move_up = coeff > 0.0;
        }
        else
        {
            move_up = coeff <= 0.0;
        }

        /* Skip if colum is at already at the bound we want to move it towards. */
        if ((move_up && is_eq(ub, old_val)) || (!move_up && is_eq(lb, old_val)))
            return;

        assert(coeff != 0.0);

        /* Exact value that makes slack zero; we need
         *      rhs - a'x - aj dxj == 0
         * <=>  rhs - a'x           = aj dxj
         * <=> (rhs - a'x) / aj     = dxj
         */
        double fix_val = old_val + slack_for_row / coeff;
        assert_if_then_else(move_up, fix_val > old_val, fix_val < old_val);

        if (col < args->n_binaries + args->n_integers)
            fix_val = move_up ? ceil(fix_val) : floor(fix_val);
        fix_val = fmin(fmax(fix_val, lb), ub);

        /* score is valid only for threadIdx.x == 0 */
        score = compute_score_single_col_move_warp(model, args, {fix_val, col});

        /* Write violation to smem. */
        if (thread_idx_warp == 0)
        {
            if (score.is_lt_feas_score(best_score))
            {
                best_score = score;
                best_move = {fix_val, col};
            }
            /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
        }
        /* Sync the warp however, before continuing with the next column from this row. */
        __syncwarp();
    }
}

/* On exit, best_scores and best_random_moves contain for each block the best move and score found by the block. Consequently, best_scores and best_random_moves need to be larger than the grid dimension. */
__global__ void compute_random_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs* args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ curandState random_state[N_WARPS_PER_BLOCK];
    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];
    /* Array for drawing the sample columns of this block. */
    __shared__ int draws[N_MOVES_PER_WARP * N_WARPS_PER_BLOCK];

    /* Set random seed and best move per warp. */
    if (thread_idx_warp == 0) {
        init_curand_warp<N_WARPS_PER_BLOCK>(random_state[warp_id], config.random_seed);

        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }

    __syncthreads();

    auto [beg, end] = get_warp_sampling_range(args->ncols, config.n_samples);
    const int cols_range = end - beg;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        /* Draw this warp's column sample. */
        warp_sample_range(draws, beg, end, config.random_seed);
        const int* draws_warp = draws + warp_id * N_MOVES_PER_WARP;

        for (int move = 0; move < N_MOVES_PER_WARP; ++move)
        {
            const int col = draws_warp[move];

            if (col == -1)
                continue;

            assert(beg <= col && col < end);

            if (is_tabu(args->tabu, col, args->iter, args->tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_random_move(model, random_state[warp_id], args, col, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

/* On exit, best_scores and best oneopt move (greedy or feasible) contain for each block the best move and score found by the block. Consequently, best_scores and best_oneopt_moves need to be larger than the grid dimension.
TODO: specialize for n_moves >= n_cols */
template <const bool GREEDY>
__global__ void compute_oneopt_moves_kernel(const GpuModelPtrs model, const TabuSearchKernelArgs *args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];
    /* Array for drawing the sample columns of this block. */
    __shared__ int draws[N_MOVES_PER_WARP * N_WARPS_PER_BLOCK];

    /* Set random seed and best move per warp. */
    if (thread_idx_warp == 0) {
        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }

    __syncthreads();

    auto [beg, end] = get_warp_sampling_range(args->ncols, config.n_samples);

    const int cols_range = end - beg;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        /* Draw this warp's column sample. */
        warp_sample_range(draws, beg, end, config.random_seed);
        int* draws_warp = draws + warp_id * N_MOVES_PER_WARP;

        for (int move = 0; move < N_MOVES_PER_WARP; ++move)
        {
            const int col = draws_warp[move];

            if (col == -1)
                continue;

            assert(beg <= col && col < end);

            if (is_tabu(args->tabu, col, args->iter, args->tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_oneopt_move<GREEDY>(model, args, col, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_flip_moves_kernel(const GpuModelPtrs model, const TabuSearchKernelArgs *args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];
    /* Array for drawing the sample columns of this block. */
    __shared__ int draws[N_MOVES_PER_WARP * N_WARPS_PER_BLOCK];

    /* Set random seed and best move per warp. */
    if (thread_idx_warp == 0) {
        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }

    __syncthreads();

    auto [beg, end] = get_warp_sampling_range(args->ncols, config.n_samples);

    const int cols_range = end - beg;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        /* Draw this warp's column sample. */
        warp_sample_range(draws, beg, end, config.random_seed);
        const int* draws_warp = draws + warp_id * N_MOVES_PER_WARP;

        for (int move = 0; move < N_MOVES_PER_WARP; ++move)
        {
            const int col = draws_warp[move];

            if (col == -1)
                continue;

            assert(beg <= col && col < end);

            if (is_tabu(args->tabu, col, args->iter, args->tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_flip_move(model, args, col, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_mtm_sat_moves_kernel(const GpuModelPtrs model, const TabuSearchKernelArgs *args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];
    /* Array for drawing the sample columns of this block. */
    __shared__ int draws[N_MOVES_PER_WARP * N_WARPS_PER_BLOCK];

    /* Set random seed and best move per warp. */
    if (thread_idx_warp == 0) {
        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }

    __syncthreads();

    const int n_feasible = args->nrows - args->n_violated;
    auto [beg, end] = get_warp_sampling_range(n_feasible, config.n_samples);

    const int row_range = end - beg;

    /* Need at least one row! */
    if (row_range > 0)
    {
        /* Draw this warp's row sample. */
        warp_sample_range(draws, beg, end, config.random_seed);
        const int* draws_warp = draws + warp_id * N_MOVES_PER_WARP;

        for (int move = 0; move < N_MOVES_PER_WARP; ++move)
        {
            const int row = draws_warp[move];

            if (row == -1)
                continue;

            assert(beg <= row && row < end);

            /* Compute a move for all columns in the picked row. */
            compute_mtm_move(model, args, row, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_mtm_unsat_moves_kernel(const GpuModelPtrs model, const TabuSearchKernelArgs *args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];
    /* Array for drawing the sample columns of this block. */
    __shared__ int draws[N_MOVES_PER_WARP * N_WARPS_PER_BLOCK];

    /* Set random seed and best move per warp. */
    if (thread_idx_warp == 0) {
        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }

    __syncthreads();

    auto [beg, end] = get_warp_sampling_range(args->n_violated, config.n_samples);

    const int row_range = end - beg;

    /* Need at least one row! */
    if (row_range > 0)
    {
        /* Draw this warp's row sample. */
        warp_sample_range(draws, beg, end, config.random_seed);
        const int* draws_warp = draws + warp_id * N_MOVES_PER_WARP;

        for (int move = 0; move < N_MOVES_PER_WARP; ++move)
        {
            const int row = draws_warp[move];

            if (row == -1)
                continue;

            assert(beg <= row && row < end);

            /* Compute a move for all columns in the picked row. */
            compute_mtm_move(model, args, row, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void update_weights_kernel(TabuSearchKernelArgs *args, const bool smoothing)
{
    const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx >= args->nrows)
        return;

    // Check if constraint is violated
    const double slack = args->slacks[row_idx];
    const bool is_eq = row_idx < args->n_equalities;
    const bool is_violated = is_eq ? !is_eq_feas(slack, 0) : is_lt_feas(slack, 0);

    if (smoothing) {
        // Smooth phase: decrease weights on satisfied constraints
        if (!is_violated && args->constraint_weights[row_idx] > 0) {
            args->constraint_weights[row_idx] -= 1.0;
        } else  if (is_violated) {
            // Penalize phase: increase weights on violated constraints
            args->constraint_weights[row_idx] += 1.0;
        }
    } else {
        // Monotone: always increase weights on violated constraints
        if (is_violated) {
            args->constraint_weights[row_idx] += 1.0;
        }
    }

    // TODO: this is always false
    // Special handling for objective (when feasible found)
    if (row_idx == 0 && args->is_found_feasible) {
        if (args->n_violated == 0) {
            // Increase objective weight when all constraints satisfied
            args->objective_weight += 1.0;
        }
    }
}

__global__ void apply_move(const GpuModelPtrs model, TabuSearchKernelArgs *args, const single_col_move *best_move,
    const move_score* best_score)
{
    const int thread_idx = threadIdx.x;
    const double val = best_move->val;
    const int col = best_move->col;

    const double old_val = args->current_sol[col];
    assert(model.lb[col] <= val && val <= model.ub[col]);
    assert_if_then(col < args->n_binaries + args->n_integers, is_integer(val));
    assert(!is_eq(old_val, val));

    assert(!is_tabu(args->tabu, col, args->iter, args->tabu_tenure));

    /* Iterate column and apply changes in slack. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* slack = rhs - Ax */
        args->slacks[row_idx] += (old_val - val) * coef;
    }

    if (thread_idx == 0) {
        args->tabu[col] = args->iter;
        args->current_sol[col] = val;

        ++args->iter;
        args->objective += best_score->objective_change;
        args->sum_viol += best_score->violation_change;
    }
}

/* Check whether curren_sol can be stored as new best solution in this solution stream (args->best_sol). */
__global__ void check_update_best_sol(TabuSearchKernelArgs *args) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    /* Do nothing if the current iterate is not feasbile. */
    if (!is_zero_feas(args->sum_viol))
        return;

    /* Do nothing if the current iterate is not better then our stored solution. */
    if (args->is_found_feasible && is_gt(args->objective, args->best_objective))
        return;

    for (int jcol = thread_id; jcol < args->ncols; jcol += stride)
        args->best_sol[jcol] = args->current_sol[jcol];

    if (thread_id == 0) {
        args->best_objective = args->objective;
        args->best_violation = args->sum_viol;
        args->is_found_feasible = true;
    }
}

__global__ void csr_spmv_kernel(
    const int nrows,
    const GpuModelPtrs model,
    double* __restrict__ sol,
    const double alpha,
    double* __restrict__ y)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= nrows) return;

    double sum = 0.0;

    const int inz_start = model.row_ptr[row];
    const int inz_end   = model.row_ptr[row+1];

    for (int inz = inz_start; inz < inz_end; ++inz)
        sum += model.row_val[inz] * sol[model.col_idx[inz]];

    y[row] += alpha * sum;
}

/* Count all elements of vector where pred(vector[i]) == true. Adds result on device to res_device. */
__global__ void count_violated_kernel(TabuSearchKernelArgs *args_device)
{
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    /* Do not early abort, BlockReduce assumes all threads of the block participate. */
    int count_thread = 0;

    for (int i = thread_idx; i < args_device->nrows; i += stride)
    {
        const int is_eq = i < args_device->n_equalities;
        const double slack_row = args_device->slacks[i];

        /* If we have an equality, any slack counts. If we have an inequality, we only count negative slack rhs - Ax >= 0. Branching does not really matter here. */
        const bool is_violated = is_eq ? !is_eq_feas(slack_row, 0) : is_lt_feas(slack_row, 0);

        if (is_violated)
            ++count_thread;
    }

    /* Block wide reduce of count_thread. */
    using BlockReduce = cub::BlockReduce<int, dense_row_col_kernels_blocksize>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    const int count_block = BlockReduce(temp_storage).Sum(count_thread);

    /* Thread 0 of each block adds to res_device atomically. So use threadIdx.x instead of thread_idx. */
    if (threadIdx.x == 0)
    {
        atomicAdd(&(args_device->n_violated), count_block);
    }
}

double thrust_dot_product(const thrust::device_vector<double> &a,
                          const thrust::device_vector<double> &b, const cudaStream_t stream)
{
    // Check sizes
    assert(a.size() == b.size());

    // Compute dot product: sum(a[i] * b[i])
    return thrust::inner_product(
        thrust::cuda::par.on(stream),
        a.begin(), a.end(),
        b.begin(),
        0.0 /* Init value */
    );
}

struct CompareViolatedFirst
{
    const TabuSearchKernelArgs* args;

    __device__ bool operator()(int i1, int i2) const
    {
        const int is_eq1 = i1 < args->n_equalities;
        const int is_eq2 = i2 < args->n_equalities;

        const double slack_row1 = args->slacks[i1];
        const double slack_row2 = args->slacks[i2];

        /* If we have an equality, any slack counts. If we have an inequality, we only count negative slack rhs - Ax >= 0. Branching does not really matter here. */
        const bool is_viol1 = is_eq1 ? !is_eq_feas(slack_row1, 0) : is_lt_feas(slack_row1, 0);
        const bool is_viol2 = is_eq2 ? !is_eq_feas(slack_row2, 0) : is_lt_feas(slack_row2, 0);

        /* For both violated return the smaller index, else return the violated constraint. */
        return (is_viol1 && is_viol2) ? (i1 < i2) : (is_viol1 ? i1 : i2);
    }
};

/* Given n_samples, the amount of total to be sampled moves, assign a fraction of total samples to each class of moves according to probabilities. A move can never have 0 samples but will always contain at least one assigned sample. */
std::array<int, AVAILABLE_MOVES> distribute_samples(const int n_samples, const moves_probability &probabilities)
{
    std::array<int,AVAILABLE_MOVES> out{};
    std::array<double, AVAILABLE_MOVES> exact{};
    std::array<double, AVAILABLE_MOVES> frac{};

    FP_ASSERT(AVAILABLE_MOVES <= n_samples);

    /* Assign one each. */
    for (int i = 0; i < AVAILABLE_MOVES; ++i) {
        out[i] = 1;
    }

    int assigned = AVAILABLE_MOVES;
    const int remaining_samples = n_samples - assigned;

    /* Compute exact allocations and round down for now. Count the amount of total assigned samples and the fractionality of each rounded assignment. */
    for (int i = 0; i < AVAILABLE_MOVES; ++i) {
        exact[i] = probabilities[i] * static_cast<double>(remaining_samples);
        const int n_add = static_cast<int>(std::floor(exact[i]));

        out[i] += n_add;
        frac[i] = exact[i] - out[i];
        assigned += n_add;
    }

    /* Now, distribute the remainders from highest do lowest fractionality. */
    int remaining = n_samples - assigned;

    /* This loop might be expensive for many remaining or large amounts of moves. But for our purposes it should be fine. */
    while (remaining > 0) {
        int best = 0;
        for (int i = 1; i < AVAILABLE_MOVES; ++i) {
            if (frac[i] > frac[best]) {
                best = i;
            }
        }
        out[best] += 1;
        frac[best] = 0.0;
        --remaining;
    }

    return out;
}

/** Prepare the next submission round. Assign total samples to individual moves and setup random seed and result arrays for each move kernel.
 *
 * Returns {blocks_per_move, config_per_move, n_blocks_total}.
 */
std::tuple<std::array<int, AVAILABLE_MOVES>, std::array<move_config, AVAILABLE_MOVES>, int> prepare_sample_submission (thrust::device_vector<move_score>& best_scores_single_col, thrust::device_vector<single_col_move>& best_single_col_moves, moves_probability& probabilities, int& seed, int nmoves_total) {
    std::array<move_config, AVAILABLE_MOVES> config_per_move;
    std::array<int, AVAILABLE_MOVES> blocks_per_move = {};

    int n_blocks_total = 0;
    // TODO: update probabilities distribution

    /* Distribute all samples among our moves given the probabilities. */
    const auto& samples = distribute_samples(nmoves_total, probabilities);

    /* Compute total required blocks and required blocks per move. */
    for (int i = 0; i < AVAILABLE_MOVES; ++i) {
        config_per_move[i].n_samples = samples[i];
        blocks_per_move[i] = blocks_for_samples(samples[i]);

        FP_ASSERT(blocks_per_move[i] > 0);

        n_blocks_total += blocks_per_move[i];
    }

    /* Potentiall resize the result arrays and THEN, distribute them among the move kernels. */
    if (static_cast<int>(best_scores_single_col.size()) < n_blocks_total) {
        best_scores_single_col.resize(n_blocks_total);
        best_single_col_moves.resize(n_blocks_total);

        const move_score def;
        thrust::fill(best_scores_single_col.begin(), best_scores_single_col.end(), def);
    }

    move_score* best_scores_single_col_ptr = thrust::raw_pointer_cast(best_scores_single_col.data());
    single_col_move* best_single_col_moves_ptr = thrust::raw_pointer_cast(best_single_col_moves.data());

    config_per_move[0].random_seed = seed;
    config_per_move[0].best_score = best_scores_single_col_ptr;
    config_per_move[0].best_move = best_single_col_moves_ptr;

    for (int move = 1; move < AVAILABLE_MOVES; ++move) {
        const int blocks_last = blocks_per_move[move - 1];

        /* We assign seeds per submitted block and warp per block. */
        seed += blocks_last * N_WARPS_PER_BLOCK;
        best_scores_single_col_ptr += blocks_last;
        best_single_col_moves_ptr += blocks_last;

        config_per_move[move].best_score = best_scores_single_col_ptr;
        config_per_move[move].best_move = best_single_col_moves_ptr;
        config_per_move[move].random_seed = seed;
    }
    seed += blocks_per_move[AVAILABLE_MOVES - 1] * N_WARPS_PER_BLOCK;

    return {blocks_per_move, config_per_move, n_blocks_total};
}

/* Calling this synchronizes the stream as we are using thrust reduce to host. */
void EvolutionSearch::recompute_solution_metrics(int solution_index, bool reset) {
    auto& data_device = data_devices[solution_index];
    auto& args_device = args_devices[solution_index];
    const auto &sol_stream = data_device.streams.front();

    /* calculate slack */
    thrust::copy(thrust::cuda::par.on(sol_stream), model_device.rhs.begin(), model_device.rhs.end(), data_device.slacks.begin());
    CHECK_CUDA(csr_spmv_kernel<<<512, 256, 0, sol_stream>>>(model_device.nrows, gpu_model_ptrs, thrust::raw_pointer_cast(data_device.current_sol.data()), -1.0, thrust::raw_pointer_cast(data_device.slacks.data())));

    /* calculate violations for equations */
    // TODO: this is highly inefficient and synchornizes the whole stream! Ideally, we store the result direcly into args_device->sum_viol.
    // Use CUB maybe?
    const double sum_viol_eq = thrust::transform_reduce(
        thrust::cuda::par.on(sol_stream),
        data_device.slacks.begin(),
        data_device.slacks.begin() + model_host.n_equalities,
        [] __device__ (const double x) -> double {
            return !is_eq_feas(x, 0) ? fabs(x) : 0.0;
        },
        0.0,
        cuda::std::plus<double>()
    );

    /* calculate violations for ineq*/
    // TODO: this is highly inefficient and synchornizes the whole stream! Ideally, we store the result direcly into args_device->sum_viol.
    const double sum_viol_total = thrust::transform_reduce(
        thrust::cuda::par.on(sol_stream),
        data_device.slacks.begin() + model_host.n_equalities,
        data_device.slacks.end(),
        [] __device__ (const double x) -> double {
            return is_lt_feas(x, 0) ? fabs(x) : 0.0;
        },
        sum_viol_eq,
        cuda::std::plus<double>()
    );

    CHECK_CUDA(cudaMemcpyAutoAsync(&(args_device->sum_viol), &sum_viol_total, sol_stream));

    /* update objective */
    // TODO: this is highly inefficient and synchornizes the whole stream! Ideally, we store the result direcly into args_device->objective.
    const double objective = thrust_dot_product(data_device.current_sol, model_device.objective, sol_stream);

    CHECK_CUDA(cudaMemcpyAutoAsync(&(args_device->objective), &objective, sol_stream));

    /* if necessary reset weights and tabu list */
    if (reset) {
        thrust::fill(thrust::cuda::par.on(sol_stream), data_device.tabu.begin(), data_device.tabu.end(), -tabu_tenure);
        thrust::fill(thrust::cuda::par.on(sol_stream), data_device.constraint_weights.begin(), data_device.constraint_weights.end(), 1);

        constexpr double one = 1;
        CHECK_CUDA(cudaMemcpyAutoAsync(&(args_device->objective_weight), &one, sol_stream));
    }

    consoleLog("\tSol{} : Initial solution metrics viol: {} obj {}", solution_index, sum_viol_total, objective);
}

template <const bool GRAPH_ENABLED>
void EvolutionSearch::recompute_solution_violation_metrics(int solution_index)
{
    const int zero = 0;
    auto& data_device = data_devices[solution_index];
    auto& args_device = args_devices[solution_index];
    const auto sol_stream = data_device.streams.front();

    thrust::sort(thrust::cuda::par.on(sol_stream),
                 data_device.violated_constraints.begin(),
                 data_device.violated_constraints.end(),
                 CompareViolatedFirst{args_device});
    CHECK_CUDA(cudaMemcpyAutoAsync(&(args_device->n_violated), &zero, sol_stream));

    CHECK_CUDA(count_violated_kernel<<<n_blocks_dense_row_kernels, dense_row_col_kernels_blocksize, 0, sol_stream>>>(args_device));

    if (!GRAPH_ENABLED)
    {
        int n_violated;
        CHECK_CUDA(cudaMemcpyAuto(&n_violated, &(args_device->n_violated)));

        consoleLog("\tSol{} : has {} violated constraints", solution_index, n_violated);
    }
}

void EvolutionSearch::load_initial_solutions(
    const int solution_index,
    const double init_value,
    const bool restrict_to_lb,
    const bool restrict_to_ub
) {
    auto& data_device = data_devices[solution_index];
    auto& args_device = args_devices[solution_index];
    const auto sol_stream = data_device.streams.front();

    assert(!active_solutions[solution_index]);

    // initialize
    thrust::fill(thrust::cuda::par.on(sol_stream), data_device.current_sol.begin(), data_device.current_sol.end(), init_value);

    // clamp with lower bound if needed
    if (restrict_to_lb) {
        thrust::transform(thrust::cuda::par.on(sol_stream),
            data_device.current_sol.begin(), data_device.current_sol.end(),
            model_device.lb.begin(),
            data_device.current_sol.begin(),
            cuda::maximum<double>()
        );
    }

    // clamp with upper bound if needed
    if (restrict_to_ub) {
        thrust::transform(thrust::cuda::par.on(sol_stream),
            data_device.current_sol.begin(), data_device.current_sol.end(),
            model_device.ub.begin(),
            data_device.current_sol.begin(),
            cuda::minimum<double>()
        );
    }

    active_solutions[solution_index] = true;

    recompute_solution_metrics(solution_index, true);
}

template <typename RoundOp>
void EvolutionSearch::load_lp_solution(const MIPData& data, const int solution_index, RoundOp round_op)
{
    auto& data_device = data_devices[solution_index];
    auto& args_device = args_devices[solution_index];
    const auto sol_stream = data_device.streams.front();

    assert(!active_solutions[solution_index]);

    copy_host_to_device(data.primals, data_device.current_sol, sol_stream);

    thrust::transform(thrust::cuda::par.on(sol_stream),
        data_device.current_sol.begin(),
        data_device.current_sol.begin() + model_host.n_binaries + model_host.n_integers,
        data_device.current_sol.begin(), round_op
    );

    active_solutions[solution_index] = true;

    recompute_solution_metrics(solution_index, true);
}

/* Load the given solution into the evolution search pool at solution_index. */
void EvolutionSearch::load_primal_solution(const int solution_index, const std::vector<double> &sol)
{
    auto &data_device = data_devices[solution_index];
    auto &args_device = args_devices[solution_index];
    const auto sol_stream = data_device.streams.front();

    assert(!active_solutions[solution_index]);

    copy_host_to_device(sol, data_device.current_sol, sol_stream);

    active_solutions[solution_index] = true;

    recompute_solution_metrics(solution_index, true);
}

void EvolutionSearch::launch_move_kernels_to_stream(
    int solution_index,
    const std::array<int, AVAILABLE_MOVES> &blocks_per_move,
    const std::array<move_config, AVAILABLE_MOVES> &config_per_move
)
{
    auto& data_device = data_devices[solution_index];
    auto& args_device = args_devices[solution_index];

    FP_ASSERT(blocks_per_move[0] > 0);
    CHECK_CUDA(compute_random_moves_kernel<<<blocks_per_move[0], BLOCKSIZE_MOVE, 0, data_device.streams[0]>>>(
        gpu_model_ptrs, args_device, config_per_move[0]));

    FP_ASSERT(blocks_per_move[1] > 0);
    CHECK_CUDA(compute_oneopt_moves_kernel<false><<<blocks_per_move[1], BLOCKSIZE_MOVE, 0, data_device.streams[1]>>>(
        gpu_model_ptrs, args_device, config_per_move[1]));

    FP_ASSERT(blocks_per_move[2] > 0);
    CHECK_CUDA(compute_oneopt_moves_kernel<true><<<blocks_per_move[2], BLOCKSIZE_MOVE, 0, data_device.streams[2]>>>(
        gpu_model_ptrs, args_device, config_per_move[2]));

    FP_ASSERT(blocks_per_move[3] > 0);
    CHECK_CUDA(compute_flip_moves_kernel<<<blocks_per_move[3], BLOCKSIZE_MOVE, 0, data_device.streams[3]>>>(
        gpu_model_ptrs, args_device, config_per_move[3]));

    FP_ASSERT(blocks_per_move[4] > 0);
    CHECK_CUDA(compute_mtm_unsat_moves_kernel<<<blocks_per_move[4], BLOCKSIZE_MOVE, 0, data_device.streams[4]>>>(
        gpu_model_ptrs, args_device, config_per_move[4]));

    FP_ASSERT(blocks_per_move[5] > 0);
    CHECK_CUDA(compute_mtm_sat_moves_kernel<<<blocks_per_move[5], BLOCKSIZE_MOVE, 0, data_device.streams[5]>>>(
        gpu_model_ptrs, args_device, config_per_move[5]));

    for (auto &stream : data_device.streams)
    {
        cudaStreamSynchronize(stream);
    }
}

/* Return empty, active solution slot. Returns -1 if there is no empty slot. */
int EvolutionSearch::getSolutionSlot() const {
    for (int i = 0; i < active_solutions.size(); ++i) {
        if (!active_solutions[i])
            return i;
    }
    return -1;
}

/* Method synchronizes the stream. */
std::unique_ptr<Solution> make_sol_from_thrust_vector(const MIPInstance &mip, thrust::device_vector<double> x, const double obj_val, const bool isFeas, const double violation, const cudaStream_t stream)
{
    auto sol = std::make_unique<Solution>();
    sol->x.resize(mip.ncols);

    copy_device_to_host(x, sol->x, stream);

    sol->objval = obj_val;
    sol->isFeas = isFeas;
    sol->absViolation = violation;
    sol->relViolation = violation / mip.maxRhs;
    sol->foundBy = "EvoSearch";

    cudaStreamSynchronize(stream);

    FP_ASSERT(sol->isFeas == isSolFeasible(mip, sol->x));
    FP_ASSERT(equal(sol->objval, evalObj(mip, sol->x)));

    return sol;
}

/* Check whether it seems worth to copy the current solution back to device and pass it to FPR. */
void EvolutionSearch::try_store_partial_solution_for_fpr(MIPData& data, int solution_index)
{
    auto& data_device = data_devices[solution_index];
    auto& args_device = args_devices[solution_index];

    int is_found_feasible;
    CHECK_CUDA(cudaMemcpyAuto(&is_found_feasible, &(args_device->is_found_feasible)));

    /* We only add infeasible solutions. */
    if (is_found_feasible)
    return;

    auto& partials = data.partials;

    double objective;
    CHECK_CUDA(cudaMemcpyAuto(&objective, &(args_device->objective)));

    /* If our solution has a better objective than the best one in the pool, add it. */
    if (partials.n_sols() == 0 || partials.getSol(0).objval > objective) {
        const auto sol_stream = data_device.streams.front();
        double sum_viol;
        CHECK_CUDA(cudaMemcpyAuto(&sum_viol, &(args_device->sum_viol)));

        consoleInfo("\tSol{} : Moving solution to partial pool", solution_index);

        auto sol = make_sol_from_thrust_vector(data.mip, data_device.current_sol, objective, false, sum_viol, sol_stream);

        partials.add(std::move(sol), true);
    }
}

struct crossover_functor {
    unsigned int seed;
    double probability;

    crossover_functor(const unsigned int _seed, const double _probability) : seed(_seed), probability(_probability) {}

    __host__ __device__
    float operator()(const thrust::tuple<float, float, int>& t) const {
        const float a = thrust::get<0>(t);
        const float b = thrust::get<1>(t);
        const int idx = thrust::get<2>(t);

        if (a != b) {
            //TODO: not sure if there is a better way to generate the randoms
            thrust::default_random_engine rng(seed);
            thrust::uniform_real_distribution<double> dist(0.0, 1.0);

            // Discard idx numbers to get different random values per element
            rng.discard(idx);

            // Random selection: pick a if random < probability, else pick b
            return (dist(rng) < probability) ? a : b;
        }
        return a;  // Values are the same, return either
    }
};

void perform_crossover(thrust::device_vector<double>& result,
                       const thrust::device_vector<double>& parent1,
                       const thrust::device_vector<double>& parent2) {

    // TODO add streams.
    // Create index sequence - THIS IS REQUIRED
    thrust::counting_iterator index_begin(0);

    // Create iterator tuples - MUST INCLUDE THE INDEX
    const auto input_begin = thrust::make_zip_iterator(
        thrust::make_tuple(parent1.begin(), parent2.begin(), index_begin)
    );


    // Perform crossover
    // higher probability favors the first vector, aka parent1
    thrust::transform(input_begin,
                      input_begin + parent1.size(),
                      result.begin(),
                      crossover_functor(0, 0.8));  // Use time as seed
}

void EvolutionSearch::evict_solutions_and_crossover(const MIPData& data) {
    // TODO: this should probably be global (member?) variables
    // Flag indicating whether at least one active solution is feasible
    int found_feasible = 0;
    // Best (minimum) objective value among feasible active solutions
    auto best_objective = INFTY;

    // minimum sum of constraint violation among all active solutions
    auto best_violation = INFTY;

    int arg_best_objective = -1;

    // crossover candidate
    auto best_infeasible_objective = INFTY;
    int arg_crossover = -1;

    // TODO: change this; it should be taking the best found etc for each solution.
    for (int solution_index = 0; solution_index < max_solutions; solution_index++) {
        if (!active_solutions[solution_index])
            continue;

        const auto& args = args_devices[solution_index];

        double obj_sol;
        double viol_sol;
        int found_feasible_sol;

        CHECK_CUDA(cudaMemcpyAuto(&found_feasible_sol, &(args->is_found_feasible)));
        CHECK_CUDA(cudaMemcpyAuto(&obj_sol, &(args->objective)));
        CHECK_CUDA(cudaMemcpyAuto(&viol_sol, &(args->sum_viol)));

        found_feasible = found_feasible || found_feasible_sol;

        if (found_feasible_sol) {

            if (obj_sol < best_objective) {
                best_objective = obj_sol;
                arg_best_objective = solution_index;
            }
        }

        // TODO: the solution must not necessarily not found_feasible wrong but we might want to have a solution with a better solution value
        else if (obj_sol < best_infeasible_objective) {
            best_infeasible_objective = obj_sol;
            arg_crossover  = solution_index;
        }
        best_violation = std::min(viol_sol, best_violation);
    }

    // perform crossover
    if (arg_best_objective != -1 && arg_crossover != -1) {
        int solution_slot = getSolutionSlot();

        if (solution_slot != -1) {
            // TODO : current_sol or best sol?
            perform_crossover(data_devices[solution_slot].current_sol, data_devices[arg_best_objective].current_sol, data_devices[arg_crossover].current_sol);
            active_solutions[solution_slot] = true;
            recompute_solution_metrics(solution_slot, true);
        }
    }

    for (int solution_index = 0; solution_index < max_solutions; solution_index++) {
        if (!active_solutions[solution_index])
            continue;

        const auto& args = args_devices[solution_index];
        double viol_sol;
        CHECK_CUDA(cudaMemcpyAuto(&viol_sol, &(args->sum_viol)));

        // Case 1: At least one feasible solution exists.
        // Keep only solutions whose objective is reasonably close
        // to the best feasible objective or are infeasible but close to the being satisfied
        if (found_feasible) {

            double obj_sol;
            CHECK_CUDA(cudaMemcpyAuto(&obj_sol, &(args->objective)));

            if ((obj_sol - best_objective) / std::abs(best_objective) > 0.2 || data.mip.maxRhs * model_host.ncols * 0.2 < viol_sol) {
                active_solutions[solution_index] = false;
                consoleLog("\t Sol{}: removed", solution_index);
            }
        }
        // Case 2: No feasible solution exists yet.
        // Prune solutions whose constraint violation is significantly
        // worse than the current best violation.
        // TODO: think about the parameters
        else if (!found_feasible && std::min(data.mip.maxRhs * model_host.ncols * 0.5, best_violation * 2) < viol_sol) {
            active_solutions[solution_index] = false;
            consoleLog("\t Sol{}: removed", solution_index);
        }
    }
}

void EvolutionSearch::load_solutions_from_pool(SolutionPool& solpool, std::vector<bool>& was_sol_loaded) {
    constexpr int LOAD_N = 3;

    if (!solpool.hasFeas())
        return;

    consoleLog("Loading best {} unparsed solutions from solution pool", LOAD_N);
    const int poolsize = solpool.n_sols();

    for (int i = was_sol_loaded.size(); i < poolsize; ++i)
        was_sol_loaded.push_back(false);

    int n_loaded = 0;

    for (int iSol = 0; iSol < poolsize; ++iSol) {
        const int nth_best = solpool.getNthBestPos(iSol);

        if (was_sol_loaded[nth_best])
            continue;

        const auto& sol = solpool.getSol(nth_best);

        if (!sol.isFeas)
            continue;

        was_sol_loaded[nth_best] = true;
        const int slot = getSolutionSlot();

        // No available slot for another solution
        if (slot == -1)
            return;
        assert(!active_solutions[slot]);

        load_primal_solution(slot, sol.x);

#ifdef EXTENDED_DEBUG
        double sum_viol;
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaMemcpyAuto(&sum_viol, &(args_devices[slot]->sum_viol)));
        assert(sum_viol == 0);
#endif
        constexpr int one = 1;
        CHECK_CUDA(cudaMemcpyAutoAsync(&(args_devices[slot]->is_found_feasible), &one, data_devices[slot].streams.front()));

        ++n_loaded;
        if (n_loaded == LOAD_N)
            return;
    }

    consoleLog("Loaded {} solutions", n_loaded);
}

/* Run N iterations of the tabu search.
 *
 * FPR -> 100x (LocalMIP -> Improvement and local MIN) -> crossover + move back to FPR (potentially worse).
 *
 *  - recompute submission distribution
 *
 *  each round:
 *   - submit all move kernels
 *   - find best move
 *   - apply best move
 *   - potentially store best found solution.
 *
We do not communicate solutions during this search. Rather, when applying a move, we check whether the
 * new solution is better than the best solution found during this batch of iterations. If it is, we swap best_sol
 */
template <const bool GRAPH_ENABLED>
void EvolutionSearch::run_evolution_search_graph(int solution_index, moves_probability& probabilities, int& seed)
{
    auto& data_device = data_devices[solution_index];
    auto& args_device = args_devices[solution_index];
    const auto sol_stream = data_device.streams.front();

    /* For smoothing decision. */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto& moves = data_device.moves;
    auto& scores = data_device.move_scores;

    /* Update the kernel parameters for our graph w.r.t. solution_index. */

    /* Each kernel and block get assigned as this rounds random seed:
     * i_rounds * (MOVES * BLOCKS) + BLOCKS * i_move + i_block
     */

    /* Update the moves distribution, compute number of moves and blocks per moves kernel. This might reallocate best_scores_single_col and best_single_col_moves. Updates global seed count and assignes each kernel a unique seed. */
    const auto [blocks_per_move, config_per_move, n_blocks_total] = prepare_sample_submission(scores, moves, probabilities, seed, n_moves_total);

    // for (N_ITER)
    // bool capture = true;
    // if (capture) {

        recompute_solution_violation_metrics<GRAPH_ENABLED>(solution_index);

        /* Submits all moves on separate streams and synchronizes afterwards. */
        launch_move_kernels_to_stream(solution_index, blocks_per_move, config_per_move);

        /* Reduce best moves to get globally best move. */
        // TODO replace this with device only kernel! Use atomicMIn and cub block reduce or maybe even device reduce ?
        auto max_iter = thrust::min_element(thrust::cuda::par.on(sol_stream), scores.begin(),
                                            scores.begin() + n_blocks_total,
                                            [] __device__(const move_score &a, const move_score &b)
                                            {
                                                return a.is_lt_feas_score(b);
                                            });
        cudaStreamSynchronize(sol_stream);

    // } else {
    //     /* Update parameters according to the new distribution and launch the graph. */
    // }

#ifdef EXTENDED_DEBUG
    if (!GRAPH_ENABLED)
    {
        int n_violated;
        CHECK_CUDA(cudaMemcpyAuto(&n_violated, &(args_device->n_violated)));
        if (n_violated == 0 && solution_index <= 4)
        {
            consoleInfo("\tSol{} : Found feasible!", solution_index);
            return;
        }
    }
#endif

    int min_index = max_iter - scores.begin();
    move_score score = (*max_iter); // Hidden copy GPU -> CPU

    /* Remove if else for fixed graph topology. Make this 2 streams and have one idle. */
    if (score.weighted_violation_change >= 0.0)
    {
        consoleLog("\tSol{} : No more good moves; updating weights!", solution_index);

        /* Update the weights and continue!  */
        const bool smoothing = dist(gen) < SMOOTHING_PROBABILITY;

        CHECK_CUDA(update_weights_kernel<<<n_blocks_dense_row_kernels, dense_row_col_kernels_blocksize, 0, sol_stream>>>(args_device, smoothing));
    }
    else
    {
        double min_value = score.weighted_violation_change;
        assert(min_value != DBL_MAX && min_value < 0.0);

        int offset_random = 0;
        int offset_oneopt = offset_random + blocks_per_move[0];
        int offset_oneopt_greedy = offset_oneopt + blocks_per_move[1];
        int offset_flip = offset_oneopt_greedy + blocks_per_move[2];
        int offset_mtm_unsat = offset_flip + blocks_per_move[3];
        int offset_mtm_sat = offset_mtm_unsat + blocks_per_move[4];
#ifdef EXTENDED_DEBUG
        move_type selected_move;
        if (min_index >= offset_mtm_sat)
            selected_move = move_type::mtm_sat;
        else if (min_index >= offset_mtm_unsat)
            selected_move = move_type::mtm_unsat;
        else if (min_index >= offset_flip)
            selected_move = move_type::flip;
        else if (min_index >= offset_oneopt_greedy)
            selected_move = move_type::oneopt_greedy;
        else if (min_index >= offset_oneopt)
            selected_move = move_type::oneopt;
        else if (min_index >= offset_random)
            selected_move = move_type::random;
        else
            assert(false);

        consoleLog("\tSol{} : Taking {} move (obj_change, slack_change, score)): ({}, {}, {})", solution_index,
                   toString(selected_move), score.objective_change, score.violation_change, score.weighted_violation_change);
#endif
        /* Apply best move. */
        CHECK_CUDA(apply_move<<<1, 1024, 0, sol_stream>>>(gpu_model_ptrs, args_device,
                                               thrust::raw_pointer_cast(moves.data()) + min_index,
                                               thrust::raw_pointer_cast(scores.data()) + min_index));

        CHECK_CUDA(check_update_best_sol<<<n_blocks_dense_row_kernels, dense_row_col_kernels_blocksize, 0, sol_stream>>>(args_device));
        cudaStreamSynchronize(sol_stream);

#ifdef EXTENDED_DEBUG
        // TODO diable when graph enabled.
        double obj_sol;
        CHECK_CUDA(cudaMemcpyAuto(&obj_sol, &(args_device->objective)));
        double sum_viol;
        CHECK_CUDA(cudaMemcpyAuto(&sum_viol, &(args_device->sum_viol)));

        const double obj_recomp = thrust::inner_product(data_device.current_sol.begin(),
                                                        data_device.current_sol.end(),
                                                        model_device.objective.begin(), 0.0);
        assert(is_eq_feas(obj_recomp, obj_sol));

        /* calculate violations for equations*/
        double aux_sol_viol = thrust::transform_reduce(
            thrust::device,
            data_device.slacks.begin(),
            data_device.slacks.begin() + model_host.n_equalities,
            [] __device__(const double x) -> double
            {
                return !is_eq_feas(x, 0) ? fabs(x) : 0.0;
            },
            0.0,
            cuda::std::plus<double>());

        /* calculate violations for ineq*/
        aux_sol_viol = thrust::transform_reduce(
            thrust::device,
            data_device.slacks.begin() + model_host.n_equalities,
            data_device.slacks.end(),
            [] __device__(const double x) -> double
            {
                return is_lt_feas(x, 0) ? fabs(x) : 0.0;
            },
            aux_sol_viol,
            cuda::std::plus<double>());
        assert(is_eq_feas(aux_sol_viol, sum_viol));
#endif
    }
}

EvolutionSearch::EvolutionSearch(const MIPInstance& model_host_, const GpuModel& model_device_) : model_host(model_host_), model_device(model_device_), gpu_model_ptrs(model_device.get_ptrs()), active_solutions(max_solutions, false) {

    n_moves_total = 1e5 * AVAILABLE_MOVES;

    data_devices.reserve(max_solutions);

    /* Vector storing device pointers of kernel arguments! Needs to be freed when done. */
    args_devices.reserve(max_solutions);

    for (int i = 0; i < max_solutions; ++i) {
        data_devices.emplace_back(model_host.nrows, model_host.ncols, tabu_tenure);
        args_devices.emplace_back(create_args_and_copy_to_device(data_devices[i], model_host, tabu_tenure));
    }

    /* At most 512 but at least one block for a dense column/row kernel. */
    n_blocks_dense_row_kernels = std::min(512, (model_host.nrows + dense_row_col_kernels_blocksize - 1) / dense_row_col_kernels_blocksize);
    n_blocks_dense_column_kernels = std::min(512, (model_host.ncols + dense_row_col_kernels_blocksize - 1) / dense_row_col_kernels_blocksize);

    assert(0 < n_blocks_dense_row_kernels);
    assert(0 < n_blocks_dense_column_kernels);
}

void EvolutionSearch::run(MIPData &data) {
    /* Vector for remembering which solutions from the MIP solution pool we have already tried to parse. */
    std::vector<bool> was_sol_loaded;

    /* Graph object for submitting multiple iterations for a given solution. */
    cudaGraph_t evo_search_graph = nullptr;
    int seed = 0;

    moves_probability probabilities{};
    /* Initialize probabilities for multi-armed bandit evenly. Random moves are disabled. */
    constexpr double w = 1.0 / static_cast<double>(AVAILABLE_MOVES - 1);
    for (int i = 0; i < AVAILABLE_MOVES; ++i)
        probabilities[i] = w;
    probabilities[static_cast<int>(move_type::random)] = 0.0;

    load_initial_solutions(0, 0.0, true, true);
    load_initial_solutions(1, -MAX_VALUE_HUGE, true, false);
    load_initial_solutions(2, MAX_VALUE_HUGE, false, true);

    consoleInfo("Starting evolution search on GPU");

    /* Do some rounds:
     * get starting solutions (somehow zeros; lbs; ubs; lp_sol rounded; fpr solutions ...)
     *
     * while (true)
     *   if not loaded
     *     load lp solution
     *
     *   sometimes
     *     load solutions from pool
     *
     *   for each solution
     *     submit graph of N iterations
     *
     */

    bool lp_solution_loaded = false;
    for (int i_round = 0; i_round < n_rounds; ++i_round) {

        if ( !lp_solution_loaded && i_round % LP_SOLUTION_FREQ == 0 ) {
            /* Check whether the LP is ready yet. */
            if (data.lp_solution_ready.load(std::memory_order_acquire)) {
                load_lp_solution(data, 3, [] __host__ __device__ (const double x) { return floor(x); });
                load_lp_solution(data, 4, [] __host__ __device__ (const double x) { return ceil(x); });
                lp_solution_loaded = true;
            }
        }

        if (i_round % SOLUTION_IMPORT_FREQ == 0) {
            load_solutions_from_pool(data.solpool, was_sol_loaded);
        }

        for (int solution_index = 0; solution_index < max_solutions; ++solution_index)
        {
            if (!active_solutions[solution_index])
                continue;
            auto& args_device = args_devices[solution_index];
            auto& data_device = data_devices[solution_index];

            /* Submit (or capture on the first call) solution graph if active. If GRAPH_ENABLE == false, never runs graph but submitts everything
             * again and again for debugging. */
            run_evolution_search_graph<GRAPH_ENABLE>(solution_index, probabilities, seed);

            consoleLog("Submitted cuda graph for solution_{}", solution_index);

            if (UserBreak) {
                consoleInfo("User break; stopping evolution search");
                return;
            }
        }

        /* Check whether the graphs found new solutions. */
        for (int solution_index = 0; solution_index < max_solutions; ++solution_index) {
            if (!active_solutions[solution_index])
                continue;

            auto& args_device = args_devices[solution_index];

            /* Copy back the solution status and potentially store the solution. */
            double sum_viol;
            double objective;
            // TODO: use correct stream
            CHECK_CUDA(cudaMemcpyAuto(&sum_viol, &(args_device->sum_viol)));
            CHECK_CUDA(cudaMemcpyAuto(&objective, &(args_device->objective)));

            // TODO: check if the conclusion is correct. checkin on exactly 0 is correct, since if the constraint is with feas tol it is resetted to 0
            bool solution_turned_feasible = (sum_viol == 0);
            bool is_incumbent = solution_turned_feasible && (!data.solpool.hasFeas() || is_lt_feas(objective, data.solpool.primalBound()));

            consoleLog("\tSol{} : updated information (objective, sum_viol): {} {}", solution_index, objective, sum_viol);

            if (is_incumbent)
            {
                auto& data_device = data_devices[solution_index];

                auto sol_ptr = make_sol_from_thrust_vector(data.mip, data_device.current_sol, objective, true, sum_viol, data_device.streams.front());
                sol_ptr->timeFound = gStopWatch().elapsed();
                //TODO: add the move that generated the solution
                data.solpool.add(std::move(sol_ptr));

                consoleLog("\tSol{} feasible and submitted to Solution Pool!", solution_index);
            }
        }

        // enforce recalculation if incumbent is found or after certain rounds
        if (i_round > 0 && i_round % RECOMPUTE_SOL_METRICS_FREQ == 0)
        {
            for (int solution_index = 0; solution_index < max_solutions; ++solution_index) {
                if (!active_solutions[solution_index])
                    continue;
                recompute_solution_metrics(solution_index, false);
            }
        }

        if (i_round > 0 && i_round % SOLUTION_TRANSFER_FREQ == 0)
        {
            for (int solution_index = 0; solution_index < max_solutions; ++solution_index) {
                if (!active_solutions[solution_index])
                    continue;

                try_store_partial_solution_for_fpr(data, solution_index);
            }
        }

        if (i_round > 0 && i_round % RECOMPUTE_SOL_METRICS_FREQ == 0) {
            evict_solutions_and_crossover(data);
        }
    }

    for (const auto& device_args : args_devices) {
        cudaFree(device_args);
    }

};
