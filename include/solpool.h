#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <c++/12/map>

#include "ranker_type.h"
#include "solution.h"
#include "value_chooser_type.h"

/* Stores a pool of MIP solutions (feasible or not).
 *
 * Addresses to solutions stored in this pool stay valid throughout their lifetime. The pool stores newly found solutions in consecutive order.
 *
 * In addition, the pool maintains a solution order:
 *  - feasible first, good to bad obj
 *  - infeasible second, low to high violation
 */

enum class move_type;

struct StrategyStats {
    double bestObj = INFTY;
    size_t numFeasible = 0;
    size_t numIncumbent = 0;
};

class SolutionPool
{

protected:
    /* Vector containing the sorted solution indices. */
    std::vector<size_t> solution_rank;
    std::vector<std::unique_ptr<Solution>> pool;

    std::atomic<double> obj_cutoff;
    double objsense;
    int ncols;
    bool thread_safe;
    mutable std::mutex mtx;
    std::map<std::pair<RankerType, ValueChooserType>, StrategyStats> dfs_stats;
    std::map<move_type, StrategyStats> evo_stats;


public:
    SolutionPool() = delete;
    SolutionPool(int ncols_, double objsense_ /** 1.0 == MIN; -1.0 == MAX */, bool thread_safe);

    double get_obj_cutoff() const { return obj_cutoff.load(std::memory_order_relaxed); }

    void print_stats() const;

    /* Add solution to pool; if force == true, always add the solution. Returns position of newly added solution. */
    int add(std::unique_ptr<Solution> sol, bool force = false);

    void add(std::unique_ptr<Solution> sol, RankerType ranker, ValueChooserType chooser, bool force = false);

    void add(std::unique_ptr<Solution> sol, move_type move, bool force = false);

    /* Return a const reference to the solution at index n. The index must be valid! */
    const Solution& getSol(int idx) const;

    /* Return whether pool has feasible solutions. */
    bool hasFeas() const;

    /* Return whether the pool is non-empty. */
    bool hasSols() const;

    /* Return size of solution pool (feasible and infeasible). */
    int n_sols() const;

    /* Get best feasible solution; if !hasSols, returns empty vector. */
    Solution getIncumbent() const;

    /* Return storage position of n-th best solution. */
    int getNthBestPos(int idx) const;

    /* Return objective of incumbent. Retuns +- INFTY if !hasFeas. */
    double primalBound() const;

    /* Return minimum violation of all solutions stored. */
    double minViolation() const;

    /* Merge this pool with another solution pool. This clears other. */
    void merge(SolutionPool &other);

    /* Print solution pool info. */
    void print() const;

private:
    void lock() const;
    void unlock() const;

    /* Check for feasible solution without locking mutex. */
    bool has_feas_unsafe() const;

    /* Add solution to pool without locking mutex. If force == true, always add the solution. Returns position of newly added solution in pool. */
    int add_unsafe(std::unique_ptr<Solution> sol, bool force = false);

    struct LockGuard {
        const SolutionPool& pool;
        LockGuard(const SolutionPool& p) : pool(p) { pool.lock(); }
        ~LockGuard() { pool.unlock(); }
    };
};
