#pragma once

#include "gpu_data.cuh"
#include "mip.h"

constexpr int AVAILABLE_MOVES = 6;

constexpr int dense_row_col_kernels_blocksize = 1024;


using moves_probability = std::array<double, AVAILABLE_MOVES>;

class MIPInstance;
class SolutionPool;
class TabuSearchDataDevice;
class TabuSearchKernelArgs;

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
    double weighted_objective_change = DBL_MAX;

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

struct TabuSearchDataDevice
{
    thrust::device_vector<double> best_sol;
    thrust::device_vector<double> current_sol;
    thrust::device_vector<double> slacks;
    thrust::device_vector<int> tabu;

    thrust::device_vector<move_score> move_scores;
    thrust::device_vector<single_col_move> moves;

    /* Constraint weights initialized with 1. */
    thrust::device_vector<double> constraint_weights;

    thrust::device_vector<int> violated_constraints;

    /* Each solution keeps 6 cuda streams. Stream 0 is used for the dependent work, stream 1..5 are used to submit
     * our 6 move evaluation kernels in parallel. Should we add new kernels, then we will need to add new stream. */
    std::array<cudaStream_t, AVAILABLE_MOVES> streams;

    TabuSearchDataDevice(const int nrows_, const int ncols_, const int tabu_tenure);
    ~TabuSearchDataDevice();
};

struct TabuSearchKernelArgs
{
    /* We run the tabu search in batches, applying multiple moves in a batch of iterations without stopping.
     * To not miss extracting good a solution during this/to avoid making a good solution worse again with new moves, we store the best
     * found feasible solution in a separate buffer: best_sol with objective best_objective and violation best_violation.
     *
     * is_found_feasible is set to indicate a feasible solution is buffered with.
     * After communicating the solution to the pool, is_found_feasible, best_objective, and best_violation should be reset.
     */
    double *best_sol;
    double best_objective;
    double best_violation;
    int is_found_feasible;

    double *current_sol;
    double *slacks;
    int *tabu;

    double *constraint_weights;
    double objective_weight;

    /* Contains a partition of violated constraints first, satisfied constraints later. */
    int *violated_constraints;

    double sum_viol;
    double objective;

    int n_violated;
    int iter;

    /* Rows are sorted [equalities, inequalities] */
    int nrows;
    int n_equalities;

    /* Columns are sorted [binaries, integers, continuous] */
    int ncols;
    int n_binaries;
    int n_integers;

    int tabu_tenure;
};

class EvolutionSearch {
public:
    const MIPInstance& model_host;
    const GpuModel& model_device;
    const GpuModelPtrs gpu_model_ptrs;

    /* Solution pool data. */

    /* Tabu search data handles; lives on Host. */
    std::vector<TabuSearchDataDevice> data_devices;
    /* Tabu search arguments, lives on Device. */
    std::vector<TabuSearchKernelArgs*> args_devices;

    std::vector<bool> active_solutions;

    static constexpr int max_solutions = 10;
    int n_moves_total = 1e5;
    int n_rounds = 10000;

    /* A column is tabu if it got moved during the last n_tabu iterations. Apply move marks a column at tabu by recording the
     * current iteration in the tabu array. When computing a move, we check whether tabu[col] >= iteration - n_tabu, if so,
     * the column may not be used.
     */
    int tabu_tenure = 10;

    /* For kernels processing a single dense row or column, we always submit blocks of size dense_row_col_kernels_blocksize. The gridSize for these kernels is at most 512 but may be much lower, if the columns/rows of a given problem are short. The two following members should be considered const and are set in the EvolutionSearch constructor. */
    int n_blocks_dense_row_kernels = 512;
    int n_blocks_dense_column_kernels = 512;

    EvolutionSearch(const MIPInstance& model_host, const GpuModel& model_device);

    void run(MIPData &data);

    /* Return empty, active solution slot. Returns -1 if there is no empty slot. */
    int getSolutionSlot() const;

    /* Recomputes violation and objective of given solution. */
    void recompute_solution_metrics(int solution_index, bool reset = false);

    /* Recomputes number of violated constraints of given solution. */
    template <const bool GRAPH_ENABLED>
    void recompute_solution_violation_metrics(int solution_index);

    /* Load an initial solution to solution_index using init_value */
    void load_initial_solutions(const int solution_index, const double init_value, const bool restrict_to_lb,
        const bool restrict_to_ub);

    /* Load the given solution into the evolution search pool at solution_index. */
    void load_primal_solution(const int solution_index, const std::vector<double> &sol);

    /* Load the LP solution, potentially available in model_host, into the evolution search pool using the rounding operation round_op. */
    template<typename RoundOp>
    void load_lp_solution(const MIPData& data, const int solution_index, RoundOp round_op);

    /* Load solutions from the given CPU solution pool into the evolution search pool. */
    void load_solutions_from_pool(SolutionPool& solpool, std::vector<bool>& was_sol_loaded);

    template <const bool GRAPH_ENABLE>
    void run_evolution_search_graph(int solution_index, moves_probability& probabilities, int& seed);

    void launch_move_kernels_to_stream(int solution_index,
        const std::array<int, AVAILABLE_MOVES> &blocks_per_move,
        const std::array<move_config, AVAILABLE_MOVES> &config_per_move
    );

    /* Check whether it seems worth to copy the current solution back to device and pass it to FPR. */
    void try_store_partial_solution_for_fpr(MIPData& data, int solution_index);

    void evict_solutions_and_crossover(const MIPData& data);
};