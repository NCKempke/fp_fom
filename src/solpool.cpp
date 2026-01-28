#include "solpool.h"

#include "solution.h"
#include "tolerances.h"
#include "tool_assert.h"

#include <consolelog.h>

SolutionPool::SolutionPool(int ncols_, double objsense_, bool thread_safe_) : ncols(ncols_), objsense(objsense_), thread_safe(thread_safe_)
{
    /* Assert minimization problem. */
    FP_ASSERT(objsense == 1.0);
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

Solution SolutionPool::getSol(int idx) const {
    LockGuard lock(*this);

    if (idx < 0 || idx >= pool.size())
        return {};

    return *pool[idx];
}

std::vector<Solution> SolutionPool::getSols(int n) const
{
    LockGuard lock(*this);
    FP_ASSERT(n >= -1);

    std::vector<Solution> sols;

    if (n == -1)
        n = pool.size();
    else
        n = std::min(n, static_cast<int>(pool.size()));

    /* Copy and return all solutions. */
    sols.reserve(n);

    for (int isol = 0; isol < n; ++isol)
        sols.push_back(*pool[isol]);

    return sols;
}

bool SolutionPool::has_feas_unsafe() const {
    return ((!pool.empty()) && pool[0]->isFeas);
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
    return has_feas_unsafe() ? *pool[0] : Solution();
}

double SolutionPool::primalBound() const
{
    LockGuard lock(*this);
    return has_feas_unsafe() ? pool[0]->objval : objsense * INFTY;
}

double SolutionPool::minViolation() const
{
    LockGuard lock(*this);
    FP_ASSERT(!pool.empty());
    return pool[0]->absViolation;
}

void SolutionPool::add_unsafe(SolutionPtr sol) {
    // avoid adding duplicates
    auto end = pool.end();
    auto compEq = [&](const SolutionPtr &other)
    {
        return (*sol) == (*other);
    };

    auto itr = std::find_if(pool.begin(), pool.end(), compEq);
    if (itr != end)
        return;

    // feasible solution are kept sorted by objective and infeasible ones by violation
    // feasible solutions always come before infeasible ones
    auto comp = [&](const SolutionPtr &sol1, const SolutionPtr &sol2)
    {
        if (sol1->isFeas == sol2->isFeas)
        {
            if (sol1->isFeas)
                return (objsense * sol1->objval < objsense * sol2->objval);
            else
                return (sol1->absViolation < sol2->absViolation);
        }
        return (sol1->isFeas);
    };

    pool.insert(
        std::upper_bound(pool.begin(), pool.end(), sol, comp),
        sol);

    // 20 solutions should be enough for our purposes
    // TODO!
    if (pool.size() > 20)
        pool.resize(20);
}

void SolutionPool::merge(SolutionPool &other)
{
    LockGuard lock1(*this);
    LockGuard lock2(other);

    for (SolutionPtr sol : other.pool)
        add_unsafe(sol);

    other.pool.clear();
}

void SolutionPool::add(SolutionPtr sol)
{
    if (!sol)
        return;

    LockGuard lock(*this);

    const bool has_feas = has_feas_unsafe();

    if ((has_feas && sol->objval < pool[0]->objval) || (!has_feas && sol->isFeas))
        consoleLog("found new incumbent : {:>15.2f}{:>15.4f}{:>15.4f}{:>7}{:>8.2f}  {}",
               sol->objval, sol->relViolation, sol->absViolation, sol->isFeas, sol->timeFound, sol->foundBy);

    add_unsafe(sol);
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

    SolutionPtr first = pool[0];
    for (int k = 0; k < pool.size(); k++)
    {
        SolutionPtr sol = pool[k];
        consoleLog("{:>8}{:>15.2f}{:>15.4f}{:>15.4f}{:>7}{:>12.2f}{:>8.2f}  {}",
                   k, sol->objval, sol->relViolation, sol->absViolation, sol->isFeas, solDistance(first->x, sol->x), sol->timeFound, sol->foundBy);
    }
}