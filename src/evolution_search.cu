#include "evolution_search.cuh"

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>

#include "cub/cub.cuh"

constexpr int N_BLOCKS_RANDOMMOVE = 512;
constexpr int BLOCKSIZE_RANDOM_MOVE = 32;

struct random_move
{
    double val;
    int col;
};

/* Large scores are bad. */
__device__ inline double random_move_score(double objective, double violation)
{
    // TODO
    return objective + violation;
}

/* activities = rhs - Ax */
__device__ void compute_random_move(const GpuModel &model, curandState &state, const double *slack, const double *sol, double objective, double sum_slack, int col, double &best_score, random_move &best_move)
{
    const int block_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    if (block_idx >= col)
        return;

    assert(0 <= col && col < model.ncols);

    // TODO column must be integer
    double col_val = sol[col];
    double lb = model.lb[col];
    double ub = model.ub[col];
    double obj_coef = model.objective[col];

    __shared__ double random_val;
    if (thread_idx == 0)
    {
        random_val = curand_uniform(&state);
    }
    /* Make sure shared memory is visible to all threads in the block. */
    __syncthreads();

    const double fixval = static_cast<int>((lb + (ub - lb) * random_val) + 0.5);
    const double delta = fixval - col_val;
    const double delta_obj = delta * obj_coef;
    double slack_change_thread = 0.0;

    /* Iterate column and compute changes in violation. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += blockDim.x)
    {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];

        /* We have <= and = only. */
        const int is_eq = model.is_equality[row_idx];

        const double slack_old = slack[row_idx];
        const double slack_new = slack_old - coef * delta;

        double viol_old = is_eq * abs(slack_old) + (1 - is_eq) * max(0.0, -slack_old);
        double viol_new = is_eq * abs(slack_new) + (1 - is_eq) * max(0.0, -slack_new);

        slack_change_thread += (viol_new - viol_old);
    }

    using BlockReduce = cub::BlockReduce<double, BLOCKSIZE_RANDOM_MOVE>;

    /* Allocate shared memory for BlockReduce. */
    __shared__ typename BlockReduce::TempStorage temp_storage;

    /* Reduce all slack changes to thread 0 of this block. */
    double slack_change = BlockReduce(temp_storage).Sum(slack_change_thread);

    /* Write violation to global memory. */
    if (thread_idx == 0)
    {
        double score = random_move_score(objective + delta_obj, sum_slack + slack_change);

        if (score < best_score)
        {
            best_score = score;
            best_move = {fixval, col};
        }

        /* best_score and best_move live in smem; however, only thread_idx == 0 touches them (for now) so we don't __syncthreads here. */
    }
}

/* On exit, best_scores and best_random_moves contain for each block the best move and score found by the block. Consequently, best_scores and best_random_moves need to be larger than the grid dimension. */
__global__ void compute_random_moves_kernel(const GpuModel &model, const double *slack, const double *sol, double objective, double sum_slack, double *best_scores, random_move *best_random_moves, int n_cols, int n_moves)
{
    const int block_idx = blockIdx.x;
    const int grid_dim = gridDim.x;
    const int thread_idx = threadIdx.x;
    __shared__ curandState state;
    __shared__ random_move best_move;
    __shared__ double best_score;

    /* Initialize shared memory on thread 0. */
    if (thread_idx == 0)
    {
        /* Set random seed to 0. */
        curand_init(0, 0, 0, &state);
        best_move = {0.0, -1};
        best_score = DBL_MAX;
    }
    __syncthreads();

    int n_cols_per_block = (n_cols + grid_dim - 1) / grid_dim;
    // TODO: this is not quite exact
    int n_moves_per_block = (n_moves + grid_dim - 1) / grid_dim;

    int my_cols_start = block_idx * n_cols_per_block;
    int my_cols_end = min(n_cols, (block_idx + 1) * n_cols_per_block);
    const int cols_range = my_cols_end - my_cols_start + 1;

    assert(my_cols_start < n_cols or n_moves_per_block == 0);

    for (int move = 0; move < n_moves_per_block; ++move)
    {
        /* Pick a column in our interval. This is uniformly distributed over [my_cols_start,..,my_cols_end]. */
        const int col = my_cols_start + static_cast<int>((cols_range * curand_uniform(&state) - 0.5));

        /* Compute a move for the picked column. */
        compute_random_move(model, state, slack, sol, objective, sum_slack, col, best_score, best_move);
    }

    /* offload the best move and its score the main memory */
    if (thread_idx == 0)
    {
        best_random_moves[block_idx] = best_move;
        best_scores[block_idx] = best_score;
    }
}

void EvolutionSearch::run()
{
    thrust::device_vector<double> sol(model.ncols, 0.0);

    // TODO compute slacks, solution objective and violation
    double objective = DBL_MAX;
    double sum_slack = DBL_MAX;
    thrust::device_vector<double> slacks(model.nrows, 0.0);

    thrust::device_vector<double> best_scores(N_BLOCKS_RANDOMMOVE, 0.0);
    thrust::device_vector<random_move> best_random_moves(N_BLOCKS_RANDOMMOVE);

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

    for (int round = 0; round < n_rounds; ++round)
    {
        /* Compute best move for each block. */
        compute_random_moves_kernel<<<N_BLOCKS_RANDOMMOVE, BLOCKSIZE_RANDOM_MOVE>>>(model, thrust::raw_pointer_cast(slacks.data()), thrust::raw_pointer_cast(sol.data()), objective, sum_slack, thrust::raw_pointer_cast(best_scores.data()), thrust::raw_pointer_cast(best_random_moves.data()), model.ncols, 1e6);

        /* Reduce best moves to get globally best move. */

        /* Apply best move. */
    }
};