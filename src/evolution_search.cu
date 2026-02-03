#include "evolution_search.cuh"

#include "gpu_data.cuh"
#include "mip.h"

#include <consolelog.h>

#include <curand_kernel.h>
#include <cuda_runtime.h>
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

constexpr int N_MAX_BLOCKS_PER_MOVE = 512;    /* Maximum number of blocks used for any move kernel */
constexpr int BLOCKSIZE_VECTOR_KERNEL = 1024; /* Blocksize used for vector kernels (each thread operating on one vector element). */

constexpr int N_BLOCKS_SINGLE_COL_MOVE = 512;
constexpr int BLOCKSIZE_SINGLE_COL_MOVE = 32;
constexpr int AVAILABLE_MOVES = 5;

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

// TODO : move to utils header
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

struct TabuSearchDataDevice
{
    // Device-resident vectors
    thrust::device_vector<double> sol;
    thrust::device_vector<double> slacks;
    thrust::device_vector<int> tabu;

    /* Objective and constraint weights initialized with 1. */
    thrust::device_vector<double> constraint_weights;
    thrust::device_vector<double> objective_weight;

    thrust::device_vector<move_score> best_scores_single_col;
    thrust::device_vector<single_col_move> best_single_col_moves;
    thrust::device_vector<int> violated_constraints;

    // Constructor
    TabuSearchDataDevice(int nrows_, int ncols_, int tabu_tenure)
        : sol(ncols_, 0.0),
          slacks(nrows_, 0.0),
          tabu(ncols_, -tabu_tenure),
          constraint_weights(nrows_, 1),
          objective_weight(1, 1),
          best_scores_single_col(N_BLOCKS_SINGLE_COL_MOVE * AVAILABLE_MOVES, {DBL_MAX, DBL_MAX}),
          best_single_col_moves(N_BLOCKS_SINGLE_COL_MOVE * AVAILABLE_MOVES),
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

    move_score *best_scores_single_col;
    single_col_move *best_single_col_moves;

    /* Contains a partition of violated constraints first, satisfied constraints later. */
    const int *violated_constraints;

    double sum_slack{};
    double objective{};

    int n_violated{};
    int iter{};
    int nrows;
    int ncols;

    int tabu_tenure;

    size_t random_seed{};

    TabuSearchKernelArgs(TabuSearchDataDevice& data, int nrows_, int ncols_, int tabu_tenure_) : sol(thrust::raw_pointer_cast(data.sol.data())),
    slacks(thrust::raw_pointer_cast(data.slacks.data())),
    tabu(thrust::raw_pointer_cast(data.tabu.data())),
    constraint_weights(thrust::raw_pointer_cast(data.constraint_weights.data())),
    objective_weight(thrust::raw_pointer_cast(data.objective_weight.data())),
    best_scores_single_col(thrust::raw_pointer_cast(data.best_scores_single_col.data())),
    best_single_col_moves(thrust::raw_pointer_cast(data.best_single_col_moves.data())),
    violated_constraints(thrust::raw_pointer_cast(data.violated_constraints.data())),
    nrows(nrows_), ncols(ncols_), tabu_tenure(tabu_tenure_) {};
};

/* Returns for threadIdx.x == 0 the score after virtually applying give move. */
__device__ move_score compute_score_single_col_move(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, single_col_move move)
{
    const int thread_idx = threadIdx.x;
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

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
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

    using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_SINGLE_COL_MOVE>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ typename BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    const double slack_change = BlockReduce(temp_storage).Sum(slack_change_thread);
    const double weighted_viol_change = BlockReduce(temp_storage).Sum(weighted_viol_change_thread);

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

    using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_SINGLE_COL_MOVE>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ typename BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    const double slack_change = BlockReduce(temp_storage).Sum(slack_change_thread);

    // TODO weighted score
    return {delta_obj, slack_change, 0.0};
}

/* activities = rhs - Ax */
__device__ void compute_random_move(const GpuModelPtrs &model, curandState &random_state, const TabuSearchKernelArgs &args, int col, move_score &best_score, single_col_move &best_move, int ncols)
{
    const int thread_idx = threadIdx.x;

    if (col >= args.ncols)
        return;

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

        int r = get_random_int_block(random_state, iub - ilb);
        int fix_i = ilb + r;
        if (fix_i >= ival)
            fix_i++;

        fix_val = (double)fix_i;
    }
    else
    {
        fix_val = get_random_double_in_range_block(random_state, lb, ub);
    }

    assert(lb <= fix_val && fix_val <= ub);
    assert(fix_val != col_val || model.var_type[col] == 'C');

    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move(model, args, {fix_val, col});

    /* Write violation to global memory. */
    if (thread_idx == 0)
    {
        // printf("Compute score %g %g %g", score.objective_change, score.violation_change, score.weighted_violation_change);
        if (score.feas_score() < best_score.feas_score())
        {
            best_score = score;
            best_move = {fix_val, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* activities = rhs - Ax */
template <const bool GREEDY>
__device__ void compute_oneopt_move(const GpuModelPtrs &model, const TabuSearchKernelArgs &args, int col, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx = threadIdx.x;

    if (col >= args.ncols)
        return;

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
        for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
        {
            const double coef = model.row_val_trans[inz];
            const int row_idx = model.col_idx_trans[inz];
            const char sense = model.row_sense[row_idx];
            const char row_slack = args.slacks[row_idx];
            const int is_eq = (sense == 'E');
            const int is_objcoef_pos = (obj * coef > 0.0);
            assert(row_slack >= 0.0);

            const double scaled_slack = fabs(coef) * row_slack;

            // TODO : maybe allow infeasible solutions here and allow becoming infeasible
            // TODO: rounding is not considered yet
            stepsize = min(stepsize, (1 - is_eq) * (is_objcoef_pos * stepsize + (1 - is_objcoef_pos) * scaled_slack));
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

    const double fix_val = obj > 0.0 ? col_val - stepsize : col_val + stepsize;

    __syncthreads();
    assert(is_le(lb, fix_val) && is_le(fix_val, ub));

    /* score is valid only for threadIdx.x == 0 */
    const auto score = compute_score_single_col_move(model, args, {fix_val, col});

    /* Write violation to global memory. */
    if (thread_idx == 0)
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
    const int thread_idx = threadIdx.x;
    move_score score;
    double fix_val;

    if (col >= args.ncols || model.var_type[col] != 'B')
        return;

    fix_val = args.sol[col] > 0.5 ? 0 : 1;

    /* score is valid only for threadIdx.x == 0 */
    score = compute_score_single_col_move(model, args, {fix_val, col});

    /* Write violation to global memory. */
    if (thread_idx == 0)
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
__device__ void compute_mtm_unsat_move(const GpuModelPtrs &model, const TabuSearchKernelArgs& args, const int row, const int col_index, move_score &best_score, single_col_move &best_move)
{
    assert(row < args.nrows);
    assert(col_index < (model.row_ptr[row + 1] - model.row_ptr[row]));

    const double slack_for_row = args.slacks[row];
    const bool is_row_eq = model.row_sense[row] == 'E';
    const bool slack_is_pos = is_gt_feas(slack_for_row, 0);

    /* skip feasible constraints -> either equation with slack == 0 or inequalities with positive slack */
    const bool is_row_feas = ((is_row_eq && is_zero_feas(slack_for_row)) || slack_is_pos);

    if (is_row_feas)
        return;
    assert(slack_for_row != 0);

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

    // if (threadIdx.x == 0) {
    //     printf("col %d  row %d fixval %g old %g slack %g\n", col, row, fix_val, old_val, slack_for_row );
    // }
    /* score is valid only for threadIdx.x == 0 */
    score = compute_score_single_col_move(model, args, {fix_val, col});

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

__device__ void compute_mtm_sat_move(const GpuModelPtrs &model, const TabuSearchKernelArgs& args, const int row, const int col_index, move_score &best_score, single_col_move &best_move)
{
    const int thread_idx = threadIdx.x;

    if (row >= args.nrows)
        return;

    const double slack_for_row = args.slacks[row];
    // skip equations or unsatisfied inequalities (slack < 0)
    bool active = !(model.row_sense[row] != 'E' && is_le(slack_for_row, 0));

    move_score score;
    double fix_val;
    const int col = model.col_idx[model.row_ptr[row] + col_index];
    double coeff = model.row_val[model.row_ptr[row] + col_index];
    if (active)
    {
        const double lb = model.lb[col];
        const double ub = model.ub[col];
        const double old_val = args.sol[col];

        active = !((is_ge(coeff, 0) && is_eq(old_val, ub)) || (is_le(coeff, 0) && is_eq(lb, old_val)));
        if (active)
        {
            bool round_up = false;
            round_up = is_le(coeff, 0.0);

            assert(coeff != 0);
            // Exact value that makes slack zero
            fix_val = old_val + slack_for_row / coeff;

            if (model.var_type[col] != 'C')
                fix_val = round_up ? ceil(fix_val) : floor(fix_val);

            fix_val = fmin(fmax(fix_val, lb), ub);

            /* score is valid only for threadIdx.x == 0 */
            score = compute_score_single_col_move(model, args, {fix_val, col});
        }
    }
    /* Write violation to global memory. */
    if (thread_idx == 0 && active)
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
template <size_t KERNEL_SEQUENCE>
__global__ void compute_random_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState random_state;
    __shared__ single_col_move best_move;
    __shared__ move_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        /* Set random seed to 0. */
        init_curand_block(random_state, args.random_seed + KERNEL_SEQUENCE * N_MAX_BLOCKS_PER_MOVE);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    int n_cols = args.ncols;
    int n_cols_per_block = (n_cols + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_cols_start = min(n_cols, block_idx * n_cols_per_block);
    int my_cols_end = min(n_cols, (block_idx + 1) * n_cols_per_block);
    const int cols_range = my_cols_end - my_cols_start;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
            const int col = my_cols_start + get_random_int_block(random_state, cols_range);

            if (is_tabu(args.tabu, col, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_random_move(model, random_state, args, col, best_score, best_move, n_cols);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        args.best_single_col_moves[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_move;
        args.best_scores_single_col[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_score;
    }
}

/* On exit, best_scores and best oneopt move (greedy or feasible) contain for each block the best move and score found by the block. Consequently, best_scores and best_oneopt_moves need to be larger than the grid dimension.
TODO: specialize for n_moves >= n_cols */
template <size_t KERNEL_SEQUENCE, const bool GREEDY>
__global__ void compute_oneopt_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState random_state;
    __shared__ single_col_move best_move;
    __shared__ move_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        /* Set random seed to 0. */
        init_curand_block(random_state, args.random_seed + KERNEL_SEQUENCE * N_MAX_BLOCKS_PER_MOVE);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    int n_cols = args.ncols;
    int n_cols_per_block = (n_cols + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_cols_start = min(n_cols, block_idx * n_cols_per_block);
    int my_cols_end = min(n_cols, (block_idx + 1) * n_cols_per_block);
    const int cols_range = my_cols_end - my_cols_start;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end). */
            const int col = my_cols_start + get_random_int_block(random_state, cols_range);

            if (is_tabu(args.tabu, col, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_oneopt_move<GREEDY>(model, args, col, best_score, best_move);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        args.best_single_col_moves[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_move;
        args.best_scores_single_col[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_score;
    }
}

template <size_t KERNEL_SEQUENCE>
__global__ void compute_flip_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState random_state;
    __shared__ single_col_move best_move;
    __shared__ move_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        init_curand_block(random_state, args.random_seed + KERNEL_SEQUENCE * N_MAX_BLOCKS_PER_MOVE);

        /* Set random seed to 0. */
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    int n_cols = args.ncols;
    int n_cols_per_block = (n_cols + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_cols_start = min(n_cols, block_idx * n_cols_per_block);
    int my_cols_end = min(n_cols, (block_idx + 1) * n_cols_per_block);
    const int cols_range = my_cols_end - my_cols_start;

    /* Need at least one column! */
    if (cols_range > 0)
    {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            /* Pick a column in our interval. TODO: This is not uniformly distributed over [my_cols_start,..,my_cols_end). */
            const int col = my_cols_start + get_random_int_block(random_state, cols_range);

            if (is_tabu(args.tabu, col, args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_flip_move(model, args, col, best_score, best_move);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        args.best_single_col_moves[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_move;
        args.best_scores_single_col[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_score;
    }
}

template <size_t KERNEL_SEQUENCE>
__global__ void compute_mtm_sat_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState random_state;
    __shared__ single_col_move best_move;
    __shared__ move_score best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        init_curand_block(random_state, args.random_seed + KERNEL_SEQUENCE * N_MAX_BLOCKS_PER_MOVE);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    // int violated_count = end - valid_idx.begin();

    int n_rows = args.nrows;
    int n_rows_per_block = (n_rows + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_rows_start = min(n_rows, block_idx * n_rows_per_block);
    int my_rows_end = min(n_rows, (block_idx + 1) * n_rows_per_block);
    const int row_range = my_rows_end - my_rows_start;

    /* Need at least one row! */
    if (row_range > 0)
    {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            // TODO: make sure to pick a satisfied constraint
            /* Pick a row in our interval. TODO: This is not uniformly distributed over [my_rows_start,...,my_rows_end). */
            const int row = my_rows_start + get_random_int_block(random_state, row_range);
            const int col_index = get_random_int_block(random_state, model.row_ptr[row + 1] - model.row_ptr[row]);

            if (is_tabu(args.tabu, model.col_idx[model.row_ptr[row] + col_index], args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_mtm_sat_move(model, args, row, col_index, best_score, best_move);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        args.best_single_col_moves[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_move;
        args.best_scores_single_col[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_score;
    }
}

template <size_t KERNEL_SEQUENCE>
__global__ void compute_mtm_unsat_moves_kernel(const GpuModelPtrs model, TabuSearchKernelArgs args, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    // __shared__ curandState state;
    __shared__ single_col_move best_move;
    __shared__ move_score best_score;
    __shared__ curandState random_state;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        init_curand_block(random_state, args.random_seed + KERNEL_SEQUENCE * N_MAX_BLOCKS_PER_MOVE);
        best_move = {0.0, -1};
        best_score = {DBL_MAX, DBL_MAX, DBL_MAX};
    }
    __syncthreads();

    // int violated_count = end - valid_idx.begin();

    // int n_rows_per_block = (n_rows + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    // TODO: this needs to be fixed!
    // int my_rows_start = min(n_cols, block_idx * n_rows_per_block);
    // int my_rows_end = min(n_cols, (block_idx + 1) * n_rows_per_block);
    const int row_range = args.n_violated;

    /* Need at least one row! */
    if (row_range > 0)
    {
        for (int move = 0; move < n_moves_per_block; ++move)
        {
            /* Pick a row in our interval. This is uniformly distributed over [my_rows_start,...,my_rows_end). */
            const int row = args.violated_constraints[get_random_int_block(random_state, row_range)];
            const int col_index = get_random_int_block(random_state, model.row_ptr[row + 1] - model.row_ptr[row]);

            if (is_tabu(args.tabu, model.col_idx[model.row_ptr[row] + col_index], args.iter, args.tabu_tenure))
                continue;

            /* Compute a move for the picked column. */
            compute_mtm_unsat_move(model, args, row, col_index, best_score, best_move);
        }
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        args.best_single_col_moves[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_move;
        args.best_scores_single_col[KERNEL_SEQUENCE * N_BLOCKS_SINGLE_COL_MOVE + block_idx] = best_score;
    }
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
            curand_init(args.random_seed + row_idx * args.iter * args.nrows, 0, 0, &random_state);
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

__global__ void apply_move(const GpuModelPtrs model, TabuSearchKernelArgs args, int move_idx)
{
    const int thread_idx = threadIdx.x;
    const double val = args.best_single_col_moves[move_idx].val;
    const int col = args.best_single_col_moves[move_idx].col;
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

void EvolutionSearch::run()
{
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
        args_device.random_seed = (AVAILABLE_MOVES * N_MAX_BLOCKS_PER_MOVE) * i_round;
        int nmoves = 1e5;

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


         // for ( int i = 0; i < args_device.n_violated; i++) {
        //     consoleLog("Index: {}",data_device.violated_constraints[i]);
        //
        /* Compute best move for each block. */
        // compute_random_moves_kernel<0><<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
        //     gpu_model_ptrs, args_device, nmoves);

        compute_oneopt_moves_kernel<1, true><<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, args_device, nmoves);

        compute_flip_moves_kernel<2><<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, args_device, nmoves);

        compute_mtm_unsat_moves_kernel<3><<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, args_device, nmoves);

        compute_mtm_sat_moves_kernel<4><<<N_BLOCKS_SINGLE_COL_MOVE, BLOCKSIZE_SINGLE_COL_MOVE>>>(
            gpu_model_ptrs, args_device, nmoves);

        // /* ----- */
        // thrust::host_vector<move_score> host_scores = data_device.best_scores_single_col;
        // for ( auto &[objective, violation, weighted_violation]: host_scores) {
        //     consoleLog("{} {} {}", objective, violation, weighted_violation);
        // }

        /* Reduce best moves to get globally best move. */
        auto max_iter = thrust::min_element(thrust::device, data_device.best_scores_single_col.begin(),
                                            data_device.best_scores_single_col.end(),
                                            [] __device__(const move_score &a, const move_score &b)
                                            {
                                                return a.feas_score() < b.feas_score();
                                            });

        int min_index = max_iter - data_device.best_scores_single_col.begin();
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
        apply_move<<<1, 1024>>>(gpu_model_ptrs, args_device, + min_index);

        args_device.objective += score.objective_change;
        args_device.sum_slack += score.violation_change;

        consoleLog("(objective, sum_slack): {} {}", args_device.objective, args_device.sum_slack);
    }
};
