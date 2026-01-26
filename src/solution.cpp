#include "solution.h"

bool Solution::operator==(const Solution &other) const
{
    if (isPartial)
    {
        for (int j = 0; j < x.size(); j++)
        {
            if (!equal(x[j], other.x[j]))
                return false;
        }

        return true;
    }

    if (!equal(objval, other.objval))
        return false;

    if (!equal(absViolation, other.absViolation))
        return false;

    if (isFeas != other.isFeas)
        return false;
    if (x.size() != other.x.size())
        return false;
    for (int j = 0; j < x.size(); j++)
    {
        if (!equal(x[j], other.x[j]))
            return false;
    }
    return true;
}