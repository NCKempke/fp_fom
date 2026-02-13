#include "solpool.h"

#include "solution.h"
#include "tolerances.h"
#include "tool_assert.h"

#include <consolelog.h>

SolutionPool::SolutionPool(int ncols_, double objsense_, bool thread_safe_) : ncols(ncols_), objsense(objsense_), thread_safe(thread_safe_)
{
    /* Assert minimization problem. */
    FP_ASSERT(objsense == 1.0);
    obj_cutoff.store(objsense * INFTY);
}

void SolutionPool::lock() const
{
    if (thread_safe)
    {
        mtx.lock();
    }
}

void SolutionPool::unlock() const
{
    if (thread_safe)
    {
        mtx.unlock();
    }
}

const Solution& SolutionPool::getSol(int idx) const {
    LockGuard lock(*this);

    FP_ASSERT(0 <= idx && idx < pool.size());

    return *pool[idx];
}

bool SolutionPool::has_feas_unsafe() const {
    if (pool.empty())
        return false;

    const size_t best_sol = solution_rank[0];
    return pool[best_sol]->isFeas;
}

bool SolutionPool::hasFeas() const
{
    LockGuard lock(*this);
    return has_feas_unsafe();
}

bool SolutionPool::hasSols() const
{
    LockGuard lock(*this);
    return (!pool.empty());
}

int SolutionPool::n_sols() const {
    LockGuard lock(*this);
    return pool.size();
}

Solution SolutionPool::getIncumbent() const
{
    LockGuard lock(*this);

    if (has_feas_unsafe()) {
        const size_t best_sol = solution_rank[0];
        return *pool[best_sol];
    } else  {
        return Solution();
    }
}

int SolutionPool::getNthBestPos(int idx) const
{
    LockGuard lock(*this);
    FP_ASSERT(0 <= idx && idx < solution_rank.size());

    return solution_rank[idx];
}

double SolutionPool::primalBound() const
{
    LockGuard lock(*this);
    if (has_feas_unsafe()) {
        const size_t best_sol = solution_rank[0];
        return pool[best_sol]->objval;
    } else {
        return objsense * INFTY;
    }
}

double SolutionPool::minViolation() const
{
    LockGuard lock(*this);

    if (pool.empty())
        return INFTY;
    else {
        const size_t best_sol = solution_rank[0];
        return pool[best_sol]->absViolation;
    }
}

int SolutionPool::add_unsafe(std::unique_ptr<Solution> sol, bool force) {
    // avoid adding duplicates
    auto end = pool.end();
    auto compEq = [&](const std::unique_ptr<Solution>& other)
    {
        return (*sol) == (*other);
    };

    auto itr = std::find_if(pool.begin(), pool.end(), compEq);
    if (itr != end)
        return -1;

    const size_t new_index = pool.size();

    // Find insertion position in rank vector and add new_index
    // feasible solution are kept sorted by objective and infeasible ones by violation
    // feasible solutions always come before infeasible ones
    auto comp = [&](size_t i, const std::unique_ptr<Solution>& s) {
        const auto& existingSol = pool[i];

        if (existingSol->isFeas == s->isFeas) {
            if (existingSol->isFeas)
                return (objsense * existingSol->objval < objsense * s->objval);
            else
                return (existingSol->absViolation < s->absViolation);
        }
        return (existingSol->isFeas); // feasible comes before infeasible
    };

    // Find where to insert the new index in the rank vector
    auto insertPos = std::lower_bound(solution_rank.begin(), solution_rank.end(), sol, comp);

    if (force || std::distance(solution_rank.begin(), insertPos) < 10) {
        if (insertPos == solution_rank.begin()) {
            const double newobj = objsense * sol->objval;

            FP_ASSERT(!sol->isFeas || objsense * sol->objval < get_obj_cutoff());
            obj_cutoff.store(newobj);
        } else {
            FP_ASSERT(!sol->isFeas || objsense * sol->objval >= get_obj_cutoff());
        }

        solution_rank.insert(insertPos, new_index);
        pool.push_back(std::move(sol));
    }

    return new_index;
}

void SolutionPool::merge(SolutionPool &other)
{
    LockGuard lock1(*this);
    LockGuard lock2(other);

    for (size_t sol = 0; sol < other.pool.size(); ++sol) {
        add_unsafe(std::move(other.pool[sol]));
    }
    other.pool.clear();
}

int SolutionPool::add(std::unique_ptr<Solution> sol, bool force)
{
    if (!sol)
        return -1;

    LockGuard lock(*this);

    const bool has_feas = has_feas_unsafe();
    const double best_obj = has_feas ? pool[solution_rank[0]]->objval : objsense * INFTY;

    if ((has_feas && sol->objval < best_obj) || (!has_feas && sol->isFeas))
        consoleLog("found new incumbent : {:>15.2f}{:>15.4f}{:>15.4f}{:>7}{:>8.2f}  {}",
               sol->objval, sol->relViolation, sol->absViolation, sol->isFeas, sol->timeFound, sol->foundBy);

    return add_unsafe(std::move(sol), force);
}

static double solDistance(std::span<const double> x1, std::span<const double> x2)
{
    double ret = 0.0;
    FP_ASSERT(x1.size() == x2.size());
    for (int j = 0; j < x1.size(); j++)
    {
        ret += fabs(x1[j] - x2[j]);
    }
    return ret;
}

void SolutionPool::print() const
{
    LockGuard lock(*this);

    if (pool.empty())
    {
        consoleInfo("Empty solution pool");
        return;
    }

    consoleInfo("Solution pool: {} solutions", pool.size());
    consoleLog("{:>8}{:>15}{:>15}{:>15}{:>7}{:>12}{:>8}  {}", "n", "Objective", "RelViolation", "AbsViolation", "Feas", "L1 dist", "Time", "FoundBy");

    const size_t best_idx = solution_rank[0];
    const auto& best = *pool[best_idx];
    for (int k = 0; k < pool.size(); k++)
    {
        const int ith_sol = solution_rank[k];
        const auto& sol = *pool[ith_sol];
        consoleLog("{:>8}{:>15.2f}{:>15.4f}{:>15.4f}{:>7}{:>12.2f}{:>8.2f}  {}",
                   k, sol.objval, sol.relViolation, sol.absViolation, sol.isFeas, solDistance(best.x, sol.x), sol.timeFound, sol.foundBy);
    }
}
