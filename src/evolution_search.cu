#include "evolution_search.cuh"

#include "gpu_data.cuh"
#include "mip.h"
#include "utils.cuh"

#include <consolelog.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>

#include "cub/cub.cuh"
#include <cub/util_device.cuh>


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

constexpr int N_MOVES_PER_WARP = 16;
constexpr int N_MOVES_PER_SINGLE_COL_BLOCK = N_MOVES_PER_WARP * N_WARPS_PER_BLOCK;

constexpr int BLOCKSIZE_VECTOR_KERNEL = 1024; /* Blocksize used for vector kernels (each thread operating on one vector element). */

constexpr int N_BLOCKS_SINGLE_COL_MOVE = 512;
constexpr int AVAILABLE_MOVES = 6;

/* For multi-armed bandit maybe. */
using moves_probability = std::array<double, AVAILABLE_MOVES>;

enum class move_type {
    random = 0,
    oneopt,
    oneopt_greedy,
    flip,
    mtm_unsat,
    mtm_sat
};

const char* to_string(move_type m) {
    switch (m) {
        case move_type::random:
            return "random";
        case move_type::oneopt:
            return "oneopt";
        case move_type::oneopt_greedy:
            return "oneopt_greedy";
        case move_type::flip:
            return "flip";
        case move_type::mtm_unsat:
            return "mtm_unsat";
        case move_type::mtm_sat:
            return "mtm_sat";
        default:
            return "unknown";
    }
}

struct warp_sampling_range {
    int beg;
    int end;
    int n_samples;
};

/* For a given interval of sampling candidates [1,..,n_candidates) (e.g. rows or columns) and n_samples total to be computed samples,
 * determine for each warp in this block its assigned sampling range and its assigned number of samples. Returns {beg, end, n_samples}. */
__device__ inline warp_sampling_range get_warp_sampling_range(int n_candidates, int n_samples) {
    warp_sampling_range range{0, 0, 0};

    const int block_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    const int warp_id_block = thread_idx / WARP_SIZE;
    const int warp_id_global = block_idx * N_WARPS_PER_BLOCK + warp_id_block;

    /* Only the first n_active_warps actually compute samples. */
    const int n_active_warps = (n_samples + N_MOVES_PER_WARP - 1) / N_MOVES_PER_WARP;

    if (warp_id_global >= n_active_warps) {
        return range;
    }

    if (warp_id_global == n_active_warps - 1) {
        range.n_samples = n_samples - (n_active_warps - 1) * N_MOVES_PER_WARP;
    } else {
        range.n_samples = N_MOVES_PER_WARP;
    }
    assert(range.n_samples >= 0);

    /* Partition [1,..,n_candidates) among all active warps. */
    const int base = n_candidates / n_active_warps;
    const int remaining = n_candidates % n_active_warps;

    if (warp_id_global < remaining) {
        range.beg = warp_id_global * (base + 1);
        range.end = range.beg + (base + 1);
    } else {
        range.beg = remaining * (base + 1) + (warp_id_global - remaining) * base;
        range.end   = range.beg + base;
    }

    range.beg = min(range.beg, n_candidates);
    range.end = min(range.end, n_candidates);

    assert(range.beg <= range.end);
    assert_if_then(range.n_samples == 0, range.beg == range.end);

    return range;
}

int blocks_for_samples(int n_samples_for_type) {
    if (n_samples_for_type <= 0)
        return 0;

    return (n_samples_for_type + N_MOVES_PER_SINGLE_COL_BLOCK - 1) / N_MOVES_PER_SINGLE_COL_BLOCK;
}

/* Moves:
 * - one_opt (feas)   : push variable in direction of its objective while maintaining feasibility
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

/* Large scores are bad; we are looking for the larges score decrease (negative change is good!). */
struct move_score
{
    double objective_change = DBL_MAX;
    double violation_change = DBL_MAX;
    double weighted_violation_change = DBL_MAX;

    __host__ __device__ inline double feas_score() const
    {
        // TODO
        return /* objective +*/ weighted_violation_change;
    }
};

/* Configuration for move kernels.*/
struct move_config {
    /* Pointers to store, for each submitted block, the best found move and score. */
    move_score* best_score;
    single_col_move* best_move;

    /* How many samples to compute. */
    int n_samples{};

    /* Which random seed to use. */
    int random_seed{};
};

/* Return whether a column was marked tabu. */
__device__ inline bool is_tabu(const int *tabu_col, int col, int iter, int tabu_tenure)
{
    return tabu_col[col] > iter - tabu_tenure;
}

__device__ inline bool aspiration(const move_score &candidate,
                                  const move_score &global_best)
{
    return candidate.feas_score() < global_best.feas_score();
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
            if (best_score[warp].feas_score() < block_best_score.feas_score())
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

struct TabuSearchDataDevice
{
    // Device-resident vectors
    thrust::device_vector<double> sol;
    thrust::device_vector<double> slacks;
    thrust::device_vector<int> tabu;

    /* Objective and constraint weights initialized with 1. */
    thrust::device_vector<double> constraint_weights;
    thrust::device_vector<double> objective_weight;

    thrust::device_vector<int> violated_constraints;

    // Constructor
    TabuSearchDataDevice(int nrows_, int ncols_, int tabu_tenure)
        : sol(ncols_, 0.0),
          slacks(nrows_, 0.0),
          tabu(ncols_, -tabu_tenure),
          constraint_weights(nrows_, 1),
          objective_weight(1, 1),
          violated_constraints(nrows_) {};
};

struct TabuSearchKernelArgs
{
    double *sol;
    double *slacks;
    int *tabu;

    double *constraint_weights;
    double *objective_weight;
    bool is_found_feasible;
    double best_objective;

    /* Contains a partition of violated constraints first, satisfied constraints later. */
    const int *violated_constraints;

    double sum_slack{};
    double objective{};

    int n_violated{};
    int iter{};
    int nrows;
    int ncols;

    int tabu_tenure;

    TabuSearchKernelArgs(TabuSearchDataDevice& data, int nrows_, int ncols_, int tabu_tenure_) : sol(thrust::raw_pointer_cast(data.sol.data())),
    slacks(thrust::raw_pointer_cast(data.slacks.data())),
    tabu(thrust::raw_pointer_cast(data.tabu.data())),
    constraint_weights(thrust::raw_pointer_cast(data.constraint_weights.data())),
    objective_weight(thrust::raw_pointer_cast(data.objective_weight.data())),
    violated_constraints(thrust::raw_pointer_cast(data.violated_constraints.data())),
    nrows(nrows_), ncols(ncols_), tabu_tenure(tabu_tenure_) {};
};

/* Returns for threadIdx.x % WARP_SIZE == 0 the score after virtually applying give move. Runs on a per-warp basis and expects equal arguments across the warp. */
__device__ move_score compute_score_single_col_move_warp(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, single_col_move move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    const int col = move.col;

    const double obj_coef = model.objective[col];
    const double col_val = args.sol[col];
    const double delta = move.val - col_val;
    const double delta_obj = delta * obj_coef;
    double slack_change_thread = 0.0;
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
        const double weight = args.constraint_weights[row_idx];

        /* We have <= and = only. */
        const int is_eq = (model.row_sense[row_idx] == 'E');

        const double slack_old = args.slacks[row_idx];
        const double slack_new = slack_old - coef * delta;

        double viol_old = is_eq * abs(slack_old) + (1 - is_eq) * max(0.0, -slack_old);
        double viol_new = is_eq * abs(slack_new) + (1 - is_eq) * max(0.0, -slack_new);

        slack_change_thread += (viol_new - viol_old);
        weighted_viol_change_thread += weight * (viol_new - viol_old);
    }

    double slack_change = warp_sum_reduce(slack_change_thread);
    double weighted_viol_change = warp_sum_reduce(weighted_viol_change_thread);

    /* Objective score. */
    double obj_viol_change = 0.0;

    if (args.is_found_feasible) {
        if (delta_obj < 0) {
            obj_viol_change = args.objective_weight[0];
        } else {
            obj_viol_change = -args.objective_weight[0];
        }
    }

    return {delta_obj, slack_change, weighted_viol_change + obj_viol_change};
}

/* Returns for threadIdx.x == 0 the score after virtually applying give move. */
__device__ move_score compute_score_col_swap(const GpuModelPtrs &model, const double *slack, const double *sol, swap_move move)
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

    using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_MOVE>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ typename BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    const double slack_change = BlockReduce(temp_storage).Sum(slack_change_thread);

    // TODO weighted score
    return {delta_obj, slack_change, 0.0};
}

/* activities = rhs - Ax */
__device__ void compute_random_move(const GpuModelPtrs &model, curandState &random_state, const TabuSearchKernelArgs &args, int col, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    assert(col < args.ncols);

    double lb = model.lb[col];
    double ub = model.ub[col];

    /* Don't run if these are too close. */
    if (ub - lb < 0.001)
        return;

    double fix_val;
    double col_val = args.sol[col];

    if (model.var_type[col] == 'B')
    {
        fix_val = 1.0 - col_val;
    }
    else if (model.var_type[col] == 'I')
    {
        int ilb = (int)lb;
        int iub = (int)ub;
        int ival = (int)col_val;

        int r = get_random_int_warp(random_state, iub - ilb);
        int fix_i = ilb + r;
        if (fix_i >= ival)
            fix_i++;

        fix_val = (double)fix_i;
    }
    else
    {
        fix_val = get_random_double_in_range_warp(random_state, lb, ub);
    }

    assert(lb <= fix_val && fix_val <= ub);
    assert(fix_val != col_val || model.var_type[col] == 'C');

    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move_warp(model, args, {fix_val, col});

    /* Write violation to smem. */
    if (thread_idx_warp == 0)
    {
        // printf("Compute score %g %g %g", score.objective_change, score.violation_change, score.weighted_violation_change);
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fix_val, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx_warp == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* activities = rhs - Ax */
template <const bool GREEDY>
__device__ void compute_oneopt_move(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, int col, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    assert(col < args.ncols);

    // TODO column must be integer ?
    double col_val = args.sol[col];
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

    if (lb == -INFTY || ub == INFTY)
        return;

    if (GREEDY)
    {
        stepsize = obj > 0.0 ? col_val - lb : ub - col_val;
    }
    else
    {
        for (int inz = col_beg + thread_idx_warp; inz < col_end; inz += WARP_SIZE)
        {
            const double coef = model.row_val_trans[inz];
            const int row_idx = model.col_idx_trans[inz];
            const char sense = model.row_sense[row_idx];
            const char row_slack = args.slacks[row_idx];
            const int is_eq = (sense == 'E');
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
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fix_val, col};
        }
        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* TODO: delete flip move? This is completely equal to the binary random move. */
__device__ void compute_flip_move(const GpuModelPtrs &model, const TabuSearchKernelArgs& args, const int col, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    move_score score;
    double fix_val;

    if (col >= args.ncols || model.var_type[col] != 'B')
        return;

    fix_val = args.sol[col] > 0.5 ? 0 : 1;

    /* score is valid only for threadIdx.x == 0 */
    score = compute_score_single_col_move_warp(model, args, {fix_val, col});

    /* Write violation to smem. */
    if (thread_idx_warp == 0)
    {
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fix_val, col};
        }
        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* Compute the mtm move for an unsatisfied constraint. */
__device__ void compute_mtm_move(const GpuModelPtrs &model, const TabuSearchKernelArgs& args, const int row, const int col_index, move_score &best_score, single_col_move &best_move)
{
    int thread_idx_warp = threadIdx.x % WARP_SIZE;

    assert(row < args.nrows);
    assert(col_index < (model.row_ptr[row + 1] - model.row_ptr[row]));

    const double slack_for_row = args.slacks[row];
    const bool slack_is_pos = is_gt_feas(slack_for_row, 0);

    if (is_zero(slack_for_row))
        return;

    const int col = model.col_idx[model.row_ptr[row] + col_index];
    double coeff = model.row_val[model.row_ptr[row] + col_index];

    const double lb = model.lb[col];
    const double ub = model.ub[col];
    const double old_val = args.sol[col];

    move_score score;
    double fix_val;
    bool move_up;

    /* Try to move col as far as possible to make the constraint exactly tight/feasible; as we know the row is infeasible, move the slack rhs - a'x towards zero. */
    if (slack_is_pos)
    {
        move_up = coeff > 0.0 ? true : false;
    }
    else
    {
        move_up = coeff > 0.0 ? false : true;
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
    fix_val = old_val + slack_for_row / coeff;
    assert_if_then_else(move_up, fix_val > old_val, fix_val < old_val);

    if (model.var_type[col] != 'C')
        fix_val = move_up ? ceil(fix_val) : floor(fix_val);
    fix_val = fmin(fmax(fix_val, lb), ub);

    /* score is valid only for threadIdx.x == 0 */
    score = compute_score_single_col_move_warp(model, args, {fix_val, col});

    /* Write violation to smem. */
    if (thread_idx_warp == 0)
    {
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fix_val, col};
        }
        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* On exit, best_scores and best_random_moves contain for each block the best move and score found by the block. Consequently, best_scores and best_random_moves need to be larger than the grid dimension. */
__global__ void compute_random_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ curandState random_state[N_WARPS_PER_BLOCK];
    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];

    /* Initialize shared memory on thread 0. */
    if (thread_idx_warp == 0)
    {
        /* Set random seed to 0 on thread 0 */
        if (thread_idx_warp == 0)
            init_curand_warp<N_WARPS_PER_BLOCK>(random_state[warp_id], config.random_seed);

        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    auto [beg, end, n_samples] = get_warp_sampling_range(args.ncols, config.n_samples);
    const int cols_range = end - beg;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        for (int move = 0; move < n_samples; ++move)
        {
            /* Pick a column in our interval. This is uniformly distributed over [beg,..,end). */
            const int col = beg + get_random_int_warp(random_state[warp_id], cols_range);

            if (is_tabu(args.tabu, col, args.iter, args.tabu_tenure))
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
__global__ void compute_oneopt_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ curandState random_state[N_WARPS_PER_BLOCK];
    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];

    /* Initialize shared memory on thread 0. */
    if (thread_idx_warp == 0)
    {
        /* Set random seed to 0 on thread 0 */
        if (thread_idx_warp == 0)
            init_curand_warp<N_WARPS_PER_BLOCK>(random_state[warp_id], config.random_seed);

        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    auto [beg, end, n_samples] = get_warp_sampling_range(args.ncols, config.n_samples);
    const int cols_range = end - beg;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        for (int move = 0; move < n_samples; ++move)
        {
            /* Pick a column in our interval. This is uniformly distributed over [beg,..,end). */
            const int col = beg + get_random_int_warp(random_state[warp_id], cols_range);

            if (is_tabu(args.tabu, col, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_oneopt_move<GREEDY>(model, args, col, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_flip_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ curandState random_state[N_WARPS_PER_BLOCK];
    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];

    /* Initialize shared memory on thread 0. */
    if (thread_idx_warp == 0)
    {
        /* Set random seed to 0 on thread 0 */
        if (thread_idx_warp == 0)
            init_curand_warp<N_WARPS_PER_BLOCK>(random_state[warp_id], config.random_seed);

        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    auto [beg, end, n_samples] = get_warp_sampling_range(args.ncols, config.n_samples);
    const int cols_range = end - beg;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        for (int move = 0; move < n_samples; ++move)
        {
            /* Pick a column in our interval. TODO: This is not uniformly distributed over [beg,..,end). */
            const int col = beg + get_random_int_warp(random_state[warp_id], cols_range);

            if (is_tabu(args.tabu, col, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_flip_move(model, args, col, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_mtm_sat_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ curandState random_state[N_WARPS_PER_BLOCK];
    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];

    /* Initialize shared memory on thread 0. */
    if (thread_idx_warp == 0)
    {
        /* Set random seed to 0 on thread 0 */
        if (thread_idx_warp == 0)
            init_curand_warp<N_WARPS_PER_BLOCK>(random_state[warp_id], config.random_seed);

        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    const int n_feasible = args.nrows - args.n_violated;
    auto [beg, end, n_samples] = get_warp_sampling_range(n_feasible, config.n_samples);
    const int row_range = end - beg;

    /* Need at least one row! */
    if (row_range > 0)
    {
        for (int move = 0; move < n_samples; ++move)
        {
            /* Pick a row in our interval. TODO: This is not uniformly distributed over [beg,...,end). */
            const int row = args.violated_constraints[args.n_violated + beg + get_random_int_warp(random_state[warp_id], row_range)];
            const int col_index = get_random_int_warp(random_state[warp_id], model.row_ptr[row + 1] - model.row_ptr[row]);

            if (is_tabu(args.tabu, model.col_idx[model.row_ptr[row] + col_index], args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_mtm_move(model, args, row, col_index, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_mtm_unsat_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, move_config config)
{
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / WARP_SIZE;
    const int thread_idx_warp = thread_idx % WARP_SIZE;

    __shared__ curandState random_state[N_WARPS_PER_BLOCK];
    __shared__ single_col_move best_move[N_WARPS_PER_BLOCK];
    __shared__ move_score best_score[N_WARPS_PER_BLOCK];

    /* Initialize shared memory on thread 0. */
    if (thread_idx_warp == 0)
    {
        /* Set random seed to 0 on thread 0 */
        if (thread_idx_warp == 0)
            init_curand_warp<N_WARPS_PER_BLOCK>(random_state[warp_id], config.random_seed);

        best_move[warp_id] = {0.0, -1};
        best_score[warp_id] = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    auto [beg, end, n_samples] = get_warp_sampling_range(args.n_violated, config.n_samples);
    const int row_range = end - beg;

    /* Need at least one row! */
    if (row_range > 0)
    {
        for (int move = 0; move < n_samples; ++move)
        {
            /* Pick a row in our interval. This is uniformly distributed over [my_rows_start,...,my_rows_end). */
            const int row = args.violated_constraints[beg + get_random_int_warp(random_state[warp_id], row_range)];
            const int col_index = get_random_int_warp(random_state[warp_id], model.row_ptr[row + 1] - model.row_ptr[row]);

            if (is_tabu(args.tabu, model.col_idx[model.row_ptr[row] + col_index], args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_mtm_move(model, args, row, col_index, best_score[warp_id], best_move[warp_id]);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void update_weights_kernel(
    const GpuModelPtrs model,
    TabuSearchKernelArgs args,
    bool weight_smooth_mode,
    double smooth_prob)
{
    const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx >= args.nrows)
        return;

    // Check if constraint is violated
    const double slack = args.slacks[row_idx];
    const bool is_eq = (model.row_sense[row_idx] == 'E');
    const bool is_violated = is_eq ? !is_eq_feas(slack, 0) : is_lt_feas(slack, 0);

    if (weight_smooth_mode) {
        // Probabilistic smoothing
        __shared__ curandState random_state;
        if (threadIdx.x == 0) {
            curand_init(row_idx * args.iter * args.nrows, 0, 0, &random_state);
        }
        __syncthreads();

        float rand_val = curand_uniform(&random_state);
        bool do_smooth = (rand_val * 10000.0f) < smooth_prob;

        if (do_smooth) {
            // Smooth phase: decrease weights on satisfied constraints
            if (!is_violated && args.constraint_weights[row_idx] > 0) {
                args.constraint_weights[row_idx] -= 1.0;
            }
        } else {
            // Penalize phase: increase weights on violated constraints
            if (is_violated) {
                args.constraint_weights[row_idx] += 1.0;
            }
        }
    } else {
        // Monotone: always increase weights on violated constraints
        if (is_violated) {
            args.constraint_weights[row_idx] += 1.0;
        }
    }

    // Special handling for objective (when feasible found)
    if (row_idx == 0 && args.is_found_feasible) {
        bool all_feasible = (args.n_violated == 0);
        if (all_feasible) {
            // Increase objective weight when all constraints satisfied
            *args.objective_weight += 1.0;
        }
    }
}

__global__ void apply_move(const GpuModelPtrs model, TabuSearchKernelArgs args, single_col_move* best_move)
{
    const int thread_idx = threadIdx.x;
    const double val = best_move->val;
    const int col = best_move->col;
    const double obj = model.objective[col];

    const double old_val = args.sol[col];
    assert(model.lb[col] <= val && val <= model.ub[col]);
    assert_if_then(model.var_type[col] != 'C', is_integer(val));
    assert(!is_eq(old_val, val));

    if (thread_idx == 0)
    {
        printf("Applying move jcol %d [%g, %g], cost %g : %g -> %g\n", col, model.lb[col], model.ub[col], obj, old_val, val);
    }

    assert(!is_tabu(args.tabu, col, args.iter, args.tabu_tenure));

    /* Iterate column and apply changes in slack. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* slack = rhs - Ax */
        args.slacks[row_idx] += (old_val - val) * coef;
    }

    if (thread_idx == 0) {
        args.tabu[col] = args.iter;
        args.sol[col] = val;
    }
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

/* Use thrust to parittion into violated and non-violated constraints. */
struct IsViolated
{
    const double *slack;
    const char *row_sense;

    __device__ bool operator()(int idx) const
    {
        const int is_eq = row_sense[idx] == 'E';
        const double slack_row = slack[idx];

        /* If we have an equality, any slack counts. If we have an inequality, we only count negative slack rhs - Ax >= 0. Branching does not really matter here. */
        return is_eq ? !is_eq_feas(slack_row, 0) : is_lt_feas(slack_row, 0); // TODO: use tolerances.h when merged from other branch.
    }
};

/* Given n_samples, the amount of total to be sampled moves, assign a fraction of total samples to each class of moves according to probabilities. */
std::array<int,AVAILABLE_MOVES> distribute_samples(int n_samples, const moves_probability& probabilities)
{
    std::array<int,AVAILABLE_MOVES> out{};
    std::array<double, AVAILABLE_MOVES> exact{};
    std::array<double, AVAILABLE_MOVES> frac{};

    int assigned = 0;

    /* Compute exact allocations and round down for now. Count the amount of total assigned samples and the fractionality of each rounded assignment. */
    for (int i = 0; i < AVAILABLE_MOVES; ++i) {
        exact[i] = probabilities[i] * static_cast<double>(n_samples);
        out[i] = static_cast<int>(std::floor(exact[i]));
        frac[i] = exact[i] - out[i];
        assigned += out[i];
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

void EvolutionSearch::run()
{
    int seed = 0;

    thrust::device_vector<move_score> best_scores_single_col;
    thrust::device_vector<single_col_move> best_single_col_moves;

    moves_probability probabilities{};
    /* Initialize probabilities for mulit-armed bandit evenly. */
    const double w = 1.0 / static_cast<double>(AVAILABLE_MOVES);
    for (int i = 0; i < AVAILABLE_MOVES; ++i)
        probabilities[i] = w;

    auto gpu_model_ptrs = model_device.get_ptrs();
    TabuSearchDataDevice data_device(model_host.nrows, model_host.ncols, tabu_tenure);
    TabuSearchKernelArgs args_device(data_device, model_host.nrows, model_host.ncols, tabu_tenure);

    // TODO : move to device.
    {
        /* Initialize sol and slacks. */
        std::vector<double> sol_host(model_host.ncols, 0.0);
        std::vector<double> slacks_host = model_host.rhs;

        for (int jcol = 0; jcol < model_host.ncols; ++jcol)
            sol_host[jcol] = max(model_host.lb[jcol], min(sol_host[jcol], model_host.ub[jcol]));

        /* Compute slacks_host = slacks_host - Ax = rhs - Ax */
        model_host.rows.SpMV(-1.0, sol_host.data(), slacks_host.data());

        double sum_slack = 0.0;
        for (int irow = 0; irow < model_host.nrows; ++irow)
        {
            const double slack_row = slacks_host[irow];
            FP_ASSERT(model_host.sense[irow] == 'L' || model_host.sense[irow] == 'E');

            if (model_host.sense[irow] == 'L' && is_lt_feas(slack_row, 0))
            {
                sum_slack += fabs(slack_row);
            }
            if (model_host.sense[irow] == 'E' && !is_eq_feas(slack_row, 0))
            {
                sum_slack += fabs(slack_row);
            }
        }

        thrust::copy(slacks_host.begin(), slacks_host.end(), data_device.slacks.begin());
        thrust::copy(sol_host.begin(), sol_host.end(), data_device.sol.begin());
        args_device.sum_slack = sum_slack;
    }

    args_device.objective = thrust_dot_product(data_device.sol, model_device.objective);

    consoleInfo("Starting evolution search on GPU");

    /* Setup tabu list. A column is tabu if it got moved during the last n_tabu iterations. Apply move marks a column at tabu by
     * recording the current iteration in the tabu array. When computing a move, we check whether tabu[col] >= iteration - n_tabu,
     * if so, the column may not be used.
     */

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

    consoleLog("Initial slack     {}", args_device.sum_slack);
    consoleLog("Initial objective {}", args_device.objective);

    for (int i_round = 0; i_round < n_rounds; ++i_round)
    {
        args_device.iter = i_round;

        /* Each kernel and block get assigned as this rounds random seed:
         * i_rounds * (MOVES * BLOCKS) + BLOCKS * i_move + i_block
         */
        int nmoves_total = 1e4 * AVAILABLE_MOVES;

        thrust::sequence(data_device.violated_constraints.begin(), data_device.violated_constraints.end()); /* Set to 0,1,...,nrows-1. */

        auto partition_point = thrust::partition(
            thrust::device,
            data_device.violated_constraints.begin(),
            data_device.violated_constraints.end(),
            IsViolated{args_device.slacks, gpu_model_ptrs.row_sense});

        args_device.n_violated = partition_point - data_device.violated_constraints.begin();
        consoleLog("Found {} violated constraints", args_device.n_violated);

        if (args_device.n_violated == 0) {
            consoleInfo("Found feasible!");
            return;
        }

        /* Update the moves distribution, compute number of moves and blocks per moves kernel. This might reallocate best_scores_single_col and best_single_col_moves. Updates global seed count and assignes each kernel a unique seed. */
        const auto [blocks_per_move, config_per_move, n_blocks_total] = prepare_sample_submission(best_scores_single_col, best_single_col_moves, probabilities, seed, nmoves_total);

        // for ( int i = 0; i < args_device.n_violated; i++) {
        //     consoleLog("Index: {}",data_device.violated_constraints[i]);
        //

        move_score* best_scores_single_col_ptr = thrust::raw_pointer_cast(best_scores_single_col.data());
        single_col_move* best_single_col_moves_ptr = thrust::raw_pointer_cast(best_single_col_moves.data());

        /* Compute best move for each block. */
        compute_random_moves_kernel<<<blocks_per_move[0], BLOCKSIZE_MOVE>>>(
            gpu_model_ptrs, args_device, config_per_move[0]);

        compute_oneopt_moves_kernel<false><<<blocks_per_move[1], BLOCKSIZE_MOVE>>>(
            gpu_model_ptrs, args_device, config_per_move[1]);

        compute_oneopt_moves_kernel<true><<<blocks_per_move[2], BLOCKSIZE_MOVE>>>(
            gpu_model_ptrs, args_device, config_per_move[2]);

        compute_flip_moves_kernel<<<blocks_per_move[3], BLOCKSIZE_MOVE>>>(
            gpu_model_ptrs, args_device, config_per_move[3]);

        compute_mtm_unsat_moves_kernel<<<blocks_per_move[4], BLOCKSIZE_MOVE>>>(
            gpu_model_ptrs, args_device, config_per_move[4]);

        compute_mtm_sat_moves_kernel<<<blocks_per_move[5], BLOCKSIZE_MOVE>>>(
            gpu_model_ptrs, args_device, config_per_move[5]);

        // /* ----- */
        // thrust::host_vector<move_score> host_scores = best_scores_single_col;
        // for ( auto &[objective, violation, weighted_violation]: host_scores) {
        //     consoleLog("{} {} {}", objective, violation, weighted_violation);
        // }

        /* Reduce best moves to get globally best move. */
        auto max_iter = thrust::min_element(thrust::device, best_scores_single_col.begin(),
                                            best_scores_single_col.begin() + n_blocks_total,
                                            [] __device__(const move_score &a, const move_score &b)
                                            {
                                                return a.feas_score() < b.feas_score();
                                            });

        int min_index = max_iter - best_scores_single_col.begin();
        move_score score = (*max_iter); // Hidden copy GPU -> CPU

        if (score.feas_score() >= 0.0) {
            consoleLog("No more good moves; updating weights!");
            const int n_blocks = (model_host.nrows + BLOCKSIZE_VECTOR_KERNEL - 1) / BLOCKSIZE_VECTOR_KERNEL;
            /* Update the weigths and continue!  */
            update_weights_kernel<<<n_blocks, BLOCKSIZE_VECTOR_KERNEL>>>(gpu_model_ptrs, args_device, false, 0.0001);
            continue;
        }

        double min_value = score.feas_score();
        assert(min_value != DBL_MAX && min_value < 0.0);

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
        consoleLog("(idx, move_score: (obj_change, slack_change, score)): {} ({}, {}, {})", min_index, score.objective_change, score.violation_change, score.feas_score());

        /* Apply best move. */
        apply_move<<<1, 1024>>>(gpu_model_ptrs, args_device, thrust::raw_pointer_cast(best_single_col_moves.data()) + min_index);

        args_device.objective += score.objective_change;
        args_device.sum_slack += score.violation_change;

        consoleLog("(objective, sum_slack): {} {}", args_device.objective, args_device.sum_slack);
    }
};
