#pragma once

#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "solution.h"

/* Stores a pool of MIP solutions (feasible or not).
 *
 * Addresses to solutions stored in this pool stay valid throughout their lifetime. The pool stores newly found solutions in consecutive order.
 *
 * In addition, the pool maintains a solution order:
 *  - feasible first, good to bad obj
 *  - infeasible second, low to high violation
 */
class SolutionPool
{
protected:
    /* Vector containing the sorted solution indices. */
    std::vector<size_t> solution_rank;
    std::vector<std::unique_ptr<Solution>> pool;

    double objsense;
    int ncols;
    bool thread_safe;
    mutable std::mutex mtx;

public:
    SolutionPool() = delete;
    SolutionPool(int ncols_, double objsense_ /** 1.0 == MIN; -1.0 == MAX */, bool thread_safe);

    void add(std::unique_ptr<Solution> sol);

    /* Return a const reference to the solution at index n. The index must be valid! */
    const Solution& getSol(int idx) const;

    /* Return whether pool has feasible solutions. */
    bool hasFeas() const;

    /* Return whether the pool is non-empty. */
    bool hasSols() const;

    /* Return size of solution pool (feasible and infeasible). */
    int n_sols() const;

    /* Get best feasible solution; if !hasSols, returns empty vector. */
    Solution getIncumbent(int nth_best_sol = 0) const;

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

    /* Add solution to pool without locking mutex. */
    void add_unsafe(std::unique_ptr<Solution> sol);

    struct LockGuard {
        const SolutionPool& pool;
        LockGuard(const SolutionPool& p) : pool(p) { pool.lock(); }
        ~LockGuard() { pool.unlock(); }
    };
};
