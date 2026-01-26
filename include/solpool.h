#pragma once

#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "solution.h"

/* Stores a pool of MIP solutions (feasible or not).
 *
 * If not empty, the first solution is always the best one
 */
class SolutionPool
{
protected:
    std::vector<SolutionPtr> pool;

    double objsense;
    int ncols;
    bool thread_safe;
    mutable std::mutex mtx;

public:
    SolutionPool() = delete;
    SolutionPool(int ncols_, double objsense_ /** 1.0 == MIN; -1.0 == MAX */, bool thread_safe);

    void add(SolutionPtr sol);

    /* Return a copy if the solution at index n. */
    Solution getSol(int idx) const;

    /* Return a copy of the best n solutions. If n == -1; returns all solutions. */
    std::vector<Solution> getSols(int n) const;

    /* Return whether pool has feasible solutions. */
    bool hasFeas() const;

    /* Return whether the pool is non-empty. */
    bool hasSols() const;

    /* Return size of solution pool (feasible and infeasible). */
    int n_sols() const;

    /* Get best feasible solution; if !hasSols, returns empty vector. */
    Solution getIncumbent() const;

    /* Return objective of incumbent. Retuns +- INFTY if !hasSols. */
    double primalBound() const;

    /* Return minimum violation of all solutions stored. */
    double minViolation() const;

    /* Merge this pool with another solution pool. */
    void merge(SolutionPool &other);

    /* Print solution pool info. */
    void print() const;

private:
    void lock() const;
    void unlock() const;

    /* Check for feasible solution without locking mutex. */
    bool has_feas_unsafe() const;

    /* Add solution to pool without locking mutex. */
    void add_unsafe(SolutionPtr sol);

    struct LockGuard {
        const SolutionPool& pool;
        LockGuard(const SolutionPool& p) : pool(p) { pool.lock(); }
        ~LockGuard() { pool.unlock(); }
    };
};
