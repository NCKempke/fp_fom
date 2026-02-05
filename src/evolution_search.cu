#include "evolution_search.cuh"

#include "gpu_data.cuh"
#include "mip.h"
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

#include "cub/cub.cuh"


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

constexpr int BLOCKSIZE_VECTOR_KERNEL = 1024; /* Blocksize used for vector kernels (each thread operating on one vector element). */

constexpr int AVAILABLE_MOVES = 6;
constexpr int UPDATE_FREQUENCE = 10000;

constexpr double MAX_VALUE_FOR_HUGE_BOUNDS = 1000;


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
};

/* For a given interval of sampling candidates [1,..,n_candidates) (e.g. rows or columns) and n_samples total to be computed samples,
 * determine for each warp in this block its assigned sampling range and its assigned number of samples. Returns {beg, end, n_samples}. */
__device__ inline warp_sampling_range get_warp_sampling_range(int n_candidates, int n_samples) {
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
        int global_thread_id = blockIdx.x * blockDim.x + thread_idx;
        curand_init(seed, global_thread_id, 0, &state_thread);

        int chunk_size = range / N_MOVES_PER_WARP;
        int leftover = range % N_MOVES_PER_WARP;

        /* Case 2: larger range, draw N_MOVES_PER_WARP samples in parallel and uniquely.
         * For this, we split the range into N_MOVES_PER_WARP non-overlapping intervals and each thread picks one column from its sub-range. This enforces some additional uniformness for the draw but well. */
        for (int i = thread_idx_warp; i < N_MOVES_PER_WARP; i += WARP_SIZE) {
            /* Compute this thread's sub-range [thread_beg, thread_end) for this sample. */
            int thread_chunk_size = chunk_size + (i < leftover ? 1 : 0);

            // Compute start of this thread's chunk
            int thread_beg = i * chunk_size + min(i, leftover);
            int thread_end = thread_beg + thread_chunk_size;
            assert(thread_beg < thread_end);

            draws_warp[i] = beg + thread_beg + get_random_int_thread(state_thread, thread_end - thread_beg);
            assert(beg <= draws_warp[i] && draws_warp[i] < end);
        }
    }

    __syncwarp(); /* Ensure warp is done and synchronize its changes. */
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
 * - Sync solutions from/to global solution pool;
 * - Scoring function:
 *    -- secondary score from Local-MIP?
 * - Mutate solutions in pool after n rounds
 * - use cuda graph to submit 100-ish rounds at once
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

    /* Return this <= other w.r.t. the feasibility score. */
    __host__ __device__ inline bool is_lt_feas_score(const move_score& other) const
    {
        /* Primary criterion: feasibility score. */
        if (weighted_violation_change < other.weighted_violation_change)
            return true;

        /* Primary criterion: feasibility score. */
        if (weighted_violation_change > other.weighted_violation_change)
            return false;

        /* Tiebreaker: objective change */
        return objective_change < other.objective_change;
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

struct TabuSearchDataDevice
{
    // Device-resident vectors
    thrust::device_vector<double> sol;
    thrust::device_vector<double> slacks;
    thrust::device_vector<int> tabu;
    thrust::device_vector<double> sum_viol;
    thrust::device_vector<double> objective;
    thrust::device_vector<int> n_violated;

    /* Objective and constraint weights initialized with 1. */
    thrust::device_vector<double> constraint_weights;
    thrust::device_vector<double> objective_weight;

    thrust::device_vector<int> violated_constraints;

    TabuSearchDataDevice(const int nrows_, const int ncols_, const int tabu_tenure, const int n_solutions)
        : sol(ncols_ * n_solutions, 0.0),
          slacks(nrows_ * n_solutions, 0.0),
          tabu(ncols_ * n_solutions, -tabu_tenure),
          sum_viol(n_solutions, 0.0),
          objective(n_solutions, 0.0),
          n_violated(n_solutions, 0.0),
          constraint_weights(nrows_ * n_solutions, 1),
          objective_weight(n_solutions, 1),
          violated_constraints(nrows_ * n_solutions) {
    }

    // Constructor
    ;
};

struct TabuSearchKernelArgs
{
    double *sol;
    double *slacks;
    int *tabu;

    double *constraint_weights;
    double *objective_weight;
    bool is_found_feasible = false;
    double best_objective;

    /* Contains a partition of violated constraints first, satisfied constraints later. */
    const int *violated_constraints;

    double* sum_viol;
    double* objective;

    int* n_violated;
    int iter{};

    /* Rows are sorted [equalities, inequalities] */
    int nrows;
    int n_equalities;

    /* Columns are sorted [binaries, integers, continuous] */
    int ncols;
    int n_binaries;
    int n_integers;

    int tabu_tenure;

    TabuSearchKernelArgs(TabuSearchDataDevice& data, const MIPInstance& mip, int tabu_tenure_) : sol(thrust::raw_pointer_cast(data.sol.data())),
        slacks(thrust::raw_pointer_cast(data.slacks.data())),
        tabu(thrust::raw_pointer_cast(data.tabu.data())),
        constraint_weights(thrust::raw_pointer_cast(data.constraint_weights.data())),
        objective_weight(thrust::raw_pointer_cast(data.objective_weight.data())),

        violated_constraints(thrust::raw_pointer_cast(data.violated_constraints.data())),
        sum_viol(thrust::raw_pointer_cast(data.sum_viol.data())),
        objective(thrust::raw_pointer_cast(data.objective.data())),
        n_violated(thrust::raw_pointer_cast(data.n_violated.data())),
        nrows(mip.nrows), n_equalities(mip.n_equalities), ncols(mip.ncols), n_binaries(mip.n_binaries), n_integers(mip.n_integers), tabu_tenure(tabu_tenure_) {
    };
};

/* Returns for threadIdx.x % WARP_SIZE == 0 the score after virtually applying give move. Runs on a per-warp basis and expects equal arguments across the warp. */
__device__ move_score compute_score_single_col_move_warp(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, single_col_move move, int solution_index)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    const int col = move.col;
    int scaled_col = col + solution_index * args.ncols;

    const double obj_coef = model.objective[col];
    const double old_col_val = args.sol[scaled_col];
    const double delta = move.val - old_col_val;
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
        const int scaled_row_idx = model.col_idx_trans[inz] + row_idx * solution_index;
        const double weight = args.constraint_weights[scaled_row_idx];

        /* We have <= and = only. */
        const double is_eq = row_idx < args.n_equalities;

        const double slack_old = args.slacks[scaled_row_idx];
        const double slack_new = slack_old - coef * delta;

        const double viol_old = is_eq * fabs(slack_old) + (1 - is_eq) * fmax(0.0, -slack_old);
        const double viol_new = is_eq * fabs(slack_new) + (1 - is_eq) * fmax(0.0, -slack_new);

        viol_change_thread += viol_new - viol_old;

        const bool feas_before = is_zero_feas(viol_old);
        bool feas_after = is_zero_feas(viol_new);

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

    double slack_change = warp_sum_reduce(viol_change_thread);
    double weighted_viol_change = warp_sum_reduce(weighted_viol_change_thread);

    return {delta_obj, slack_change, weighted_viol_change};
}

/* Returns for threadIdx.x == 0 the score after virtually applying give move. */
__device__ move_score compute_score_col_swap(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, const double *slack, const double *sol, const swap_move move, int solution_index)
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
        const int is_eq = row_idx < args.n_equalities;

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
        const int is_eq = row_idx < args.n_equalities;

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
__device__ void compute_random_move(const GpuModelPtrs &model, curandState &random_state, const TabuSearchKernelArgs &args, const int col, move_score &best_score, single_col_move &best_move, int solution_index)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    assert(col < args.ncols);

    double lb = model.lb[col];
    double ub = model.ub[col];

    if (lb < 1000 || 1000 < ub)
        return;

    /* Don't run if these are too close. */
    if (ub - lb < 0.001)
        return;

    double fix_val;
    int scaled_col = col + solution_index * args.ncols;
    double col_val = args.sol[scaled_col];

    if (col < args.n_binaries)
    {
        fix_val = 1.0 - col_val;
    }
    else if (col < args.n_binaries + args.n_integers)
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

    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move_warp(model, args, {fix_val, col}, solution_index);

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
__device__ void compute_oneopt_move(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, const int col, move_score &best_score, single_col_move &best_move, const int solution_index)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;
    assert(col < args.ncols);

    const int scaled_col = col + solution_index * args.ncols;

    // TODO column must be integer ?
    const double col_val = args.sol[scaled_col];
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
            const int scaled_row_idx = row_idx + solution_index * args.nrows;
            const int is_eq = row_idx < args.n_equalities;
            const double row_slack = args.slacks[scaled_row_idx];
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
    const auto score = compute_score_single_col_move_warp(model, args, {fix_val, col}, solution_index);

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
__device__ void compute_flip_move(const GpuModelPtrs &model, const TabuSearchKernelArgs& args, const int col, move_score &best_score, single_col_move &best_move, const int solution_index)
{
    const int thread_idx_warp = threadIdx.x % WARP_SIZE;

    /* Only for binaries. */
    if (col >= args.n_binaries)
        return;

    const int scaled_col = col + solution_index * args.ncols;

    double fix_val = args.sol[scaled_col] > 0.5 ? 0 : 1;

    /* score is valid only for threadIdx.x == 0 */
    move_score score = compute_score_single_col_move_warp(model, args, {fix_val, col}, solution_index);

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
__device__ void compute_mtm_move(const GpuModelPtrs &model, const TabuSearchKernelArgs& args, const int row, move_score &best_score, single_col_move &best_move, const int solution_index)
{
    int thread_idx_warp = threadIdx.x % WARP_SIZE;

    assert(row < args.nrows);

    const int scaled_row = row + solution_index * args.nrows;
    const double slack_for_row = args.slacks[scaled_row];
    const bool slack_is_pos = is_gt_feas(slack_for_row, 0);

    if (is_zero(slack_for_row))
        return;

    for (int inz = model.row_ptr[row]; inz < model.row_ptr[row + 1]; ++inz) {
        const int col = model.col_idx[inz];
        const int scaled_col = col + solution_index * args.ncols;

        if (is_tabu(args.tabu, col + solution_index * args.ncols, args.iter, args.tabu_tenure))
            continue;

        const double coeff = model.row_val[inz];
        const double lb = model.lb[col];
        const double ub = model.ub[col];
        const double old_val = args.sol[scaled_col];

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

        if (col < args.n_binaries + args.n_integers)
            fix_val = move_up ? ceil(fix_val) : floor(fix_val);
        fix_val = fmin(fmax(fix_val, lb), ub);

        /* score is valid only for threadIdx.x == 0 */
        score = compute_score_single_col_move_warp(model, args, {fix_val, col}, solution_index);

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
__global__ void compute_random_moves_kernel(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, move_config config, const int solution_index)
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

    auto [beg, end] = get_warp_sampling_range(args.ncols, config.n_samples);
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

            if (is_tabu(args.tabu, col + solution_index * args.ncols, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_random_move(model, random_state[warp_id], args, col, best_score[warp_id], best_move[warp_id], solution_index);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

/* On exit, best_scores and best oneopt move (greedy or feasible) contain for each block the best move and score found by the block. Consequently, best_scores and best_oneopt_moves need to be larger than the grid dimension.
TODO: specialize for n_moves >= n_cols */
template <const bool GREEDY>
__global__ void compute_oneopt_moves_kernel(const GpuModelPtrs& model, const TabuSearchKernelArgs args, move_config config, const int solution_index)
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

    auto [beg, end] = get_warp_sampling_range(args.ncols, config.n_samples);

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

            if (is_tabu(args.tabu, col + solution_index * args.ncols, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_oneopt_move<GREEDY>(model, args, col, best_score[warp_id], best_move[warp_id], solution_index);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_flip_moves_kernel(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, move_config config, const int solution_index)
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

    auto [beg, end] = get_warp_sampling_range(args.ncols, config.n_samples);

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

            if (is_tabu(args.tabu, col + solution_index * args.ncols, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_flip_move(model, args, col, best_score[warp_id], best_move[warp_id], solution_index);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_mtm_sat_moves_kernel(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, move_config config, const int solution_index)
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

    const int n_feasible = args.nrows - args.n_violated[solution_index];
    auto [beg, end] = get_warp_sampling_range(n_feasible, config.n_samples);

    const int row_range = end - beg;

    /* Need at least one row! */
    if (row_range > 0)
    {
        /* Draw this warp's row sample. */
        warp_sample_range(draws, beg, end, config.random_seed);
        int* draws_warp = draws + warp_id * N_MOVES_PER_WARP;

        for (int move = 0; move < N_MOVES_PER_WARP; ++move)
        {
            const int row = draws_warp[move];

            if (row == -1)
                continue;

            assert(beg <= row && row < end);

            /* Compute a move for all columns in the picked row. */
            compute_mtm_move(model, args, row, best_score[warp_id], best_move[warp_id], solution_index);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

__global__ void compute_mtm_unsat_moves_kernel(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, move_config config, const int solution_index)
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

    auto [beg, end] = get_warp_sampling_range(args.n_violated[solution_index], config.n_samples);

    const int row_range = end - beg;

    /* Need at least one row! */
    if (row_range > 0)
    {
        /* Draw this warp's row sample. */
        warp_sample_range(draws, beg, end, config.random_seed);
        int* draws_warp = draws + warp_id * N_MOVES_PER_WARP;

        for (int move = 0; move < N_MOVES_PER_WARP; ++move)
        {
            const int row = draws_warp[move];

            if (row == -1)
                continue;

            assert(beg <= row && row < end);

            /* Compute a move for all columns in the picked row. */
            compute_mtm_move(model, args, row, best_score[warp_id], best_move[warp_id], solution_index);
        }
    }

    reduce_and_offload_best_score_in_block(best_score, best_move, config);
}

template <const bool SMOOTHING>
__global__ void update_weights_kernel(
    const GpuModelPtrs model,
    const TabuSearchKernelArgs& args,
    const int solution_index
)
{
    const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx >= args.nrows)
        return;

    const int row_idx_scaled = row_idx + solution_index * args.nrows;


    // Check if constraint is violated
    const double slack = args.slacks[row_idx_scaled];
    const bool is_eq = row_idx < args.n_equalities;
    const bool is_violated = is_eq ? !is_eq_feas(slack, 0) : is_lt_feas(slack, 0);

    if (SMOOTHING) {
        // Smooth phase: decrease weights on satisfied constraints
        if (!is_violated && args.constraint_weights[row_idx_scaled] > 0) {
            args.constraint_weights[row_idx_scaled] -= 1.0;
        } else  if (is_violated) {
            // Penalize phase: increase weights on violated constraints
            args.constraint_weights[row_idx_scaled] += 1.0;
        }
    } else {
        // Monotone: always increase weights on violated constraints
        if (is_violated) {
            args.constraint_weights[row_idx_scaled] += 1.0;
        }
    }

    // Special handling for objective (when feasible found)
    if (row_idx == 0 && args.is_found_feasible) {
        const bool all_feasible = args.n_violated[solution_index] == 0;
        if (all_feasible) {
            // Increase objective weight when all constraints satisfied
            args.objective_weight[solution_index] += 1.0;
        }
    }
}

__global__ void apply_move(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, single_col_move* best_move, const int solution_index, const double objective_change, const double violation_change)
{
    const int thread_idx = threadIdx.x;
    const double val = best_move->val;
    const int col = best_move->col;
    const int scaled_col = col + solution_index * args.ncols;
    const double obj = model.objective[col];

    const double old_val = args.sol[scaled_col];
    assert(model.lb[col] <= val && val <= model.ub[col]);
    assert_if_then(col < args.n_binaries + args.n_integers, is_integer(val));
    assert(!is_eq(old_val, val));

    if (thread_idx == 0) {
        printf("Applying move jcol %d [%g, %g], cost %g : %g -> %g\n", col, model.lb[col], model.ub[col], obj, old_val,
               val);
    }

    assert(!is_tabu(args.tabu, col + solution_index * args.ncols, args.iter, args.tabu_tenure));

    /* Iterate column and apply changes in slack. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int scaled_row_idx = model.col_idx_trans[inz] + solution_index * args.nrows;

        /* slack = rhs - Ax */
        args.slacks[scaled_row_idx] += (old_val - val) * coef;
    }

    if (thread_idx == 0) {
        args.tabu[scaled_col] = args.iter;
        args.sol[scaled_col] = val;
        args.objective[solution_index] += objective_change;
        args.sum_viol[solution_index] += violation_change;
    }
}

__global__ void csr_spmv_kernel(
    int nrows,
    int ncols,
    int n_solutions,
    const GpuModelPtrs model,
    double *__restrict__ sol,
    double alpha,
    double *__restrict__ y) {
    const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (linear_index >= nrows * n_solutions) return;

    int solution_idx = linear_index / nrows;
    int row = linear_index % nrows;
    double row_sum = 0.0;

    const int inz_start = model.row_ptr[row];
    const int inz_end = model.row_ptr[row + 1];

    for (int inz = inz_start; inz < inz_end; ++inz) {
        const int scaled_col = model.col_idx[inz] +  solution_idx * ncols;
        row_sum += model.row_val[inz] * sol[scaled_col];
    }

    y[linear_index] += alpha * row_sum;
}

__global__ void csr_spmv_kernel_index(
    int nrows,
    int ncols,
    int solution_index,
    const GpuModelPtrs model,
    double *__restrict__ sol,
    double alpha,
    double *__restrict__ y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= nrows) return;

    const int scaled_row = row + solution_index * nrows;
    double row_sum = 0.0;

    const int inz_start = model.row_ptr[row];
    const int inz_end = model.row_ptr[row + 1];

    for (int inz = inz_start; inz < inz_end; ++inz) {
        const int scaled_col = model.col_idx[inz] +  solution_index * ncols;
        row_sum += model.row_val[inz] * sol[scaled_col];
    }

    y[scaled_row] += alpha * row_sum;
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

/* Use thrust to partition into violated and non-violated constraints. */
struct IsViolated
{
    const double *slack;
    const int n_equalities;

    __device__ bool operator()(int idx) const
    {
        const int is_eq = idx < n_equalities;
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

    /* Potentially resize the result arrays and THEN, distribute them among the move kernels. */
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

void update_references_for_solution_index(const int solution_index, TabuSearchDataDevice &data_device,
                                          const GpuModel &model, const GpuModelPtrs &model_ptrs, const int tabu_tenure,
                                          bool reset = true) {
    thrust::copy(model.rhs.begin(), model.rhs.end(),
                 data_device.slacks.begin() + solution_index * model.nrows);
    csr_spmv_kernel_index<<<512, 1024>>>(model.nrows, model.ncols, solution_index, model_ptrs,
                                         thrust::raw_pointer_cast(data_device.sol.data()), -1.0,
                                         thrust::raw_pointer_cast(data_device.slacks.data()));

    //TODO move this too to device
    double sum_viol = 0;
    for (int irow = 0; irow < model.nrows; ++irow) {
        const int scaled_row = irow + solution_index * model.nrows;
        const double slack_row = data_device.slacks[scaled_row];

        // first equalities than inequalities
        if (irow < model.n_equalities) {
            if (!is_eq_feas(slack_row, 0))
                sum_viol += fabs(slack_row);
        } else {
            if (is_lt_feas(slack_row, 0))
                sum_viol += fabs(slack_row);
        }
    }
    data_device.sum_viol[solution_index] = sum_viol;

    data_device.objective[solution_index] = thrust::inner_product(
        data_device.sol.begin() + solution_index * model.ncols,
        data_device.sol.begin() + (solution_index + 1) * model.ncols,
        model.objective.begin(),
        0.0);
    if (reset) {
        thrust::fill(data_device.tabu.begin() + solution_index * model.ncols,
                     data_device.tabu.end() + (solution_index + 1) * model.ncols, -tabu_tenure);
        thrust::fill(data_device.constraint_weights.begin() + solution_index * model.nrows, data_device.constraint_weights.begin() + (solution_index+1) * model.nrows, 1);
        thrust::fill(data_device.objective_weight.begin() + solution_index * model.ncols,
                     data_device.objective_weight.begin() + (solution_index + 1) * model.ncols, 1);
        consoleLog("Initial slack {} \t initial objective {}\t for solution at pos {}",
                   data_device.sum_viol[solution_index], data_device.objective[solution_index], solution_index);
    }
}

void update_violated_constraints(const int solution_index, TabuSearchDataDevice& data_device, TabuSearchKernelArgs &args,
                                 const GpuModel &model) {
    thrust::sequence(data_device.violated_constraints.begin() + solution_index * model.nrows,
                     data_device.violated_constraints.begin() + (solution_index + 1) * model.nrows);

    auto start_index = data_device.violated_constraints.begin() + solution_index * model.nrows;
    auto partition_point = thrust::partition(
        thrust::device,
        start_index,
        data_device.violated_constraints.begin() + (solution_index + 1) * model.nrows,
        IsViolated{args.slacks + solution_index * model.nrows, args.n_equalities});
    data_device.n_violated[solution_index] = partition_point - start_index;
    consoleLog("Found {} violated constraints for index {}", data_device.n_violated[solution_index], solution_index);
}

void EvolutionSearch::run(MIPData &data) const {
    int seed = 0;

    /* For smoothing decision. */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double smooth_prob = 0.0001;

    thrust::device_vector<move_score> best_scores_single_col;
    thrust::device_vector<single_col_move> best_single_col_moves;

    moves_probability probabilities{};
    /* Initialize probabilities for multi-armed bandit evenly. Random moves are disabled. */
    const double w = 1.0 / static_cast<double>(AVAILABLE_MOVES - 1);
    for (int i = 0; i < AVAILABLE_MOVES; ++i)
        probabilities[i] = w;
    probabilities[static_cast<int>(move_type::random)] = 0.0;

    int max_solutions = 5;
    auto gpu_model_ptrs = model_device.get_ptrs();
    std::vector<bool> activate_solutions (max_solutions, false);

    TabuSearchDataDevice data_device(model_host.nrows, model_host.ncols, tabu_tenure, max_solutions);
    TabuSearchKernelArgs args_device(data_device, model_host, tabu_tenure);

    //initialize the solution vector with 3 solutions
    // 1. the zero vector within variable bounds
    // 2. the lower bounds
    // 3. the upper bounds
    // 1. case
    thrust::fill(data_device.sol.begin(), data_device.sol.begin() + model_host.ncols, 0.0);
    thrust::transform(
        data_device.sol.begin(), data_device.sol.begin() + model_host.ncols,
        model_device.lb.begin(),
        data_device.sol.begin(),
        cuda::maximum<double>()
    );
    thrust::transform(
        data_device.sol.begin(), data_device.sol.begin() + model_host.ncols,
        model_device.ub.begin(),
        data_device.sol.begin(),
        cuda::minimum<double>()
    );

    update_references_for_solution_index(0, data_device, model_host, gpu_model_ptrs, tabu_tenure);
    activate_solutions[0] = true;


    //2. case
    thrust::fill(data_device.sol.begin() + model_host.ncols, data_device.sol.begin() + model_host.ncols * 2, -MAX_VALUE_FOR_HUGE_BOUNDS);
    thrust::transform(
        data_device.sol.begin() + model_host.ncols, data_device.sol.begin() + model_host.ncols * 2,
        model_device.lb.begin(),
        data_device.sol.begin()+ model_host.ncols,
        cuda::maximum<double>()
    );
    update_references_for_solution_index(1, data_device, model_host, gpu_model_ptrs, tabu_tenure);
    activate_solutions[1] = true;

    std::vector<double> sol_host(model_host.ncols);
    thrust::copy(data_device.sol.begin() + model_host.ncols, data_device.sol.begin() + model_host.ncols *2, sol_host.begin());

    // 3. case
    thrust::fill(data_device.sol.begin() + model_host.ncols * 2, data_device.sol.begin() + model_host.ncols * 3, MAX_VALUE_FOR_HUGE_BOUNDS);
    thrust::transform(
        data_device.sol.begin() + model_host.ncols * 2, data_device.sol.begin() + model_host.ncols * 3,
        model_device.ub.begin(),
        data_device.sol.begin() + model_host.ncols * 2,
        cuda::minimum<double>()
    );
    update_references_for_solution_index(2, data_device, model_host, gpu_model_ptrs, tabu_tenure);
    activate_solutions[2] = true;


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

    bool is_relaxed_solution_copied = false;
    for (int i_round = 0; i_round < n_rounds; ++i_round)
    {
        args_device.iter = i_round;

        if (!is_relaxed_solution_copied) {
            if (data.lp_solution_ready) {
                int cont_variables_begin = model_host.n_binaries + model_host.n_integers;
                thrust::transform(
                    data.primals.begin(),
                    data.primals.begin() + cont_variables_begin,
                    data_device.sol.begin() + model_device.ncols * 4,
                    [] __host__ __device__ (double x) {
                        return floor(x);
                    }
                );
                thrust::copy(
                    data.primals.begin() + cont_variables_begin,
                    data.primals.end(),
                    data_device.sol.begin() + model_device.ncols * 4 + cont_variables_begin
                );
                update_references_for_solution_index(4, data_device, model_device,
                                                     gpu_model_ptrs, tabu_tenure);
                activate_solutions[4] = true;

                thrust::transform(
                    data.primals.begin(),
                    data.primals.begin() + cont_variables_begin,
                    data_device.sol.begin() + model_device.ncols * 4,
                    [] __host__ __device__ (double x) {
                        return ceil(x);
                    }
                );
                thrust::copy(
                    data.primals.begin() + cont_variables_begin,
                    data.primals.end(),
                    data_device.sol.begin() + model_device.ncols * 3 + cont_variables_begin
                );
                update_references_for_solution_index(3, data_device, model_device,
                                                     gpu_model_ptrs, tabu_tenure);
                activate_solutions[3] = true;
            }
            is_relaxed_solution_copied = true;
        }

        /* Each kernel and block get assigned as this rounds random seed:
         * i_rounds * (MOVES * BLOCKS) + BLOCKS * i_move + i_block
         */
        int nmoves_total = 1e5 * AVAILABLE_MOVES;


        /* iterate over all solutions*/
        for (int solution_index = 0; solution_index < max_solutions; ++solution_index) {
            /* skip inactive solutions*/
            if (!activate_solutions[solution_index])
                continue;

            // if (i_round != 0 && i_round % UPDATE_FREQUENCE == 0) {
            //     update_references_for_solution_index(solution_index, data_device, model_device, gpu_model_ptrs, tabu_tenure, false);
            // }
            update_violated_constraints(solution_index, data_device, args_device, model_host);
#define EXTENDED_DEBUG
#ifdef EXTENDED_DEBUG
            //TODO: this is just currently for EXTENDED_DEBUGGING to be removed later
            if (data_device.n_violated[solution_index] == 0) {
                consoleInfo("Found feasible!");
                return;
            }
#endif

            /* Update the moves distribution, compute number of moves and blocks per moves kernel. This might reallocate best_scores_single_col and best_single_col_moves. Updates global seed count and assignes each kernel a unique seed. */
            //TODO: make this dynamic for each solution
            const auto [blocks_per_move, config_per_move, n_blocks_total] = prepare_sample_submission(
                best_scores_single_col, best_single_col_moves, probabilities, seed, nmoves_total);

            // for ( int i = 0; i < args_device.n_violated; i++) {
            //     consoleLog("Index: {}",data_device.violated_constraints[i]);
            //

            move_score* best_scores_single_col_ptr = thrust::raw_pointer_cast(best_scores_single_col.data());
            single_col_move* best_single_col_moves_ptr = thrust::raw_pointer_cast(best_single_col_moves.data());

            if (blocks_per_move[0] > 0) {
                compute_random_moves_kernel<<<blocks_per_move[0], BLOCKSIZE_MOVE>>>(
                    gpu_model_ptrs, args_device, config_per_move[0], solution_index);
            }

            if (blocks_per_move[1] > 0) {
                compute_oneopt_moves_kernel<false><<<blocks_per_move[1], BLOCKSIZE_MOVE>>>(
                    gpu_model_ptrs, args_device, config_per_move[1], solution_index);
            }

            if (blocks_per_move[2] > 0) {
                compute_oneopt_moves_kernel<true><<<blocks_per_move[2], BLOCKSIZE_MOVE>>>(
                    gpu_model_ptrs, args_device, config_per_move[2], solution_index);
            }

            if (blocks_per_move[3] > 0) {
                compute_flip_moves_kernel<<<blocks_per_move[3], BLOCKSIZE_MOVE>>>(
                    gpu_model_ptrs, args_device, config_per_move[3], solution_index);
            }

            if (blocks_per_move[4] > 0) {
                compute_mtm_unsat_moves_kernel<<<blocks_per_move[4], BLOCKSIZE_MOVE>>>(
                    gpu_model_ptrs, args_device, config_per_move[4], solution_index);
            }

            if (blocks_per_move[5] > 0) {
                compute_mtm_sat_moves_kernel<<<blocks_per_move[5], BLOCKSIZE_MOVE>>>(
                    gpu_model_ptrs, args_device, config_per_move[5], solution_index);
            }

            // /* ----- */
            // thrust::host_vector<move_score> host_scores = best_scores_single_col;
            // for ( auto &[objective, violation, weighted_violation]: host_scores) {
            //     consoleLog("{} {} {}", objective, violation, weighted_violation);
            // }
            assert(best_scores_single_col.size() >= n_blocks_total);

            /* Reduce best moves to get globally best move. */
            auto max_iter = thrust::min_element(thrust::device, best_scores_single_col.begin(),
                                                best_scores_single_col.begin() + n_blocks_total,
                                                [] __device__(const move_score &a, const move_score &b)
                                                {
                                                    return a.is_lt_feas_score(b);
                                                });

            int min_index = max_iter - best_scores_single_col.begin();
            move_score score = (*max_iter); // Hidden copy GPU -> CPU

            if (score.weighted_violation_change >= 0.0) {
                consoleLog("No more good moves; updating weights!");
                const int n_blocks = (model_host.nrows + BLOCKSIZE_VECTOR_KERNEL - 1) / BLOCKSIZE_VECTOR_KERNEL;

                /* Update the weights and continue!  */
                if (dist(gen) < smooth_prob)
                    update_weights_kernel<true><<<n_blocks, BLOCKSIZE_VECTOR_KERNEL>>>(gpu_model_ptrs, args_device, solution_index);
                else
                    update_weights_kernel<false><<<n_blocks, BLOCKSIZE_VECTOR_KERNEL>>>(gpu_model_ptrs, args_device, solution_index);

                continue;
            }

            double min_value = score.weighted_violation_change;
            assert(min_value != DBL_MAX && min_value < 0.0);

            int offset_random = 0;
            int offset_oneopt = offset_random + blocks_per_move[0];
            int offset_oneopt_greedy = offset_oneopt + blocks_per_move[1];
            int offset_flip = offset_oneopt_greedy + blocks_per_move[2];
            int offset_mtm_unsat = offset_flip + blocks_per_move[3];
            int offset_mtm_sat = offset_mtm_unsat + blocks_per_move[4];

            std::string move_name;
            if (min_index >= offset_mtm_sat)
                move_name = "mtm_sat";
            else if (min_index >= offset_mtm_unsat)
                move_name = "mtm_unsat";
            else if (min_index >= offset_flip)
                move_name = "flip";
            else if (min_index >= offset_oneopt_greedy)
                move_name = "oneopt_greedy";
            else if (min_index >= offset_oneopt)
                move_name = "oneopt";
            else if (min_index >= offset_random)
                move_name = "random";
            else
                move_name = "unknown";

            consoleLog("[{}-sol] Taking {} move", solution_index , move_name);
            consoleLog("[{}-sol] (idx, move_score: (obj_change, slack_change, score)): {} ({}, {}, {})", solution_index, min_index, score.objective_change, score.violation_change, score.weighted_violation_change);

            /* Apply best move.  update also objective and violation*/
            apply_move<<<1, 1024>>>(gpu_model_ptrs, args_device, thrust::raw_pointer_cast(best_single_col_moves.data()) + min_index, solution_index, score.objective_change, score.violation_change);

#ifdef EXTENDED_DEBUG
            double new_objective, new_violation;

            cudaMemcpy(&new_objective,
                       args_device.objective + solution_index,
                       sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&new_violation,
                       args_device.sum_viol + solution_index,
                       sizeof(double),
                       cudaMemcpyDeviceToHost);

            consoleLog("[{}-sol] (objective change, sum_viol change): {} {}", solution_index, score.objective_change, score.violation_change);
            consoleLog("[{}-sol] (objective, sum_viol): {}  {}", solution_index, new_objective, new_violation);

            assert(is_eq_feas(thrust::inner_product( data_device.sol.begin() + solution_index * model_host.ncols,
                    data_device.sol.begin() + (solution_index + 1) * model_host.ncols,
                    model_device.objective.begin(),0.0), new_objective));
            //TODO: add here asserts that the violations (sum_slacks) are equal to the updated
#endif
        }
    }
};
