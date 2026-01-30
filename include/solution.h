#pragma once

#include <maths.h>
#include <tolerances.h>

#include <vector>

/* Stores a complete MIP solution (feasible or not) */
struct Solution
{
public:
    std::vector<double> x;

    double objval{INFTY};
    double absViolation{INFTY};
    double relViolation{INFTY};
    double timeFound{-1.0};

    std::string foundBy{}; //< which algorithm found this solution

    /* Whether this is a partial solution. For partial solutions remaining x values are set to inf; objval, violation, and iFeas have no meaning when isPartial == true. */
    bool isPartial{false};
    bool isFeas{false};

    // equality operator
    bool operator==(const Solution &other) const;
};

using SolutionPtr = std::shared_ptr<Solution>;
