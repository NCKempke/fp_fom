#include "evolution_search.cuh"
#include <curand_kernel.h>

constexpr int BLOCKSIZE_RANDOM_MOVE = 32;



/* activities = rhs - Ax */
__global__ void compute_random_move_kernel(const GpuModel& model, double objective, double sum_slack, const double* slack, const double* sol, int col) {
    const int block_id = blockIdx.x;
    const int thread_idx = threadIdx.x;

    if (block_id >= col)
        return;

    assert(0 <= col && col < model.ncols);

    // TODO column must be integer
    double col_val = sol[col];
    double lb = model.lb[col];
    double ub = model.ub[col];

    // TODO: expensive?
    curandState state;
    curand_init(clock64(), thread_idx, 0, &state);
    const double fixval = round(lb + (ub - lb) * curand_uniform(&state));
    const double delta = fix_val - col_val;

    /* Iterate column and compute changes in violation. */
    const int col_beg = model.row_ptr_trans[col];
    const int col_end = model.row_ptr_trans[col + 1];

    for (int inz = col_beg + thread_idx; inz < col_end; inz += block) {
        const double coef = model.row_val_trans[inz];
        const int row_idx = model.col_idx_trans[inz];
        const char row_sense = model.sense[inz];

        double row_slack = slack[row_idx] - coef * delta;
        const bool is_row_violated = (row_slack == 0.0 &

        if (row_sense == 'E') {
            violation -= slack + row_slack;
        } else {

        }
    }

    __syncthreads();
}


void EvolutionSearch::run() {

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

};