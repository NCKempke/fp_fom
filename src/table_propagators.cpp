/**
 * @file   table_propagators.c
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 */

#include "table_propagators.h"

#include <fmt/format.h>

#include <floats.h>

static std::string printClique(const Domain &domain, const CliqueTable &table, int cl)
{
	std::string ret = fmt::format("clq_{}: ", cl);
	for (int lit : table.getClique(cl))
	{
		const auto [var, isPos] = varFromLit(lit, domain.ncols());
		ret += fmt::format("{} {} ", isPos ? "+" : "-", domain.cNames[var]);
	}
	ret += fmt::format("{} 1", table.cliqueIsEqual(cl) ? "==" : "<=");
	return ret;
}

/* Get the direct implications of a single literal */
void CliquesPropagator::propagateOneLiteral(PropagationEngine &engine, int lit, bool &infeas)
{
	const Domain &domain = engine.getDomain();
	int n = domain.ncols();
	FP_ASSERT((lit >= 0) && (lit < 2 * n));
	const auto [var, isPos] = varFromLit(lit, n);
	FP_ASSERT(!isPos || (domain.lb(var) == 1.0));
	FP_ASSERT(isPos || (domain.ub(var) == 0.0));
	// consoleLog("Propagate literal: {} = {}", domain.cNames[var], (int)isPos);

	/* Loop over the cliques the fixed literal appears in */
	const auto &cliques = cliquetable.getLit(lit);
	for (int cl : cliques)
	{
		// consoleLog("Propagate {}", printClique(domain, cliquetable, cl));

		/* Fix all literals in the clique (except the one we are propagating) */
		const auto &clique = cliquetable.getClique(cl);
		for (int l : clique)
		{
			/* skip literal we are propagating */
			if (l == lit)
				continue;

			const auto [var, isPos] = varFromLit(l, n);

			if (isPos)
			{
				/* positive literal -> fix to zero */
				if (domain.ub(var) > 0.5)
				{
					if (domain.lb(var) > 0.5)
					{
						// consoleLog("{} = 0 inconsistent with current lb = {}", domain.cNames[var], domain.lb(var));
						infeas = true;
						break;
					}
					infeas = engine.changeUpperBound(var, 0.0);
					FP_ASSERT(!infeas);
				}
			}
			else
			{
				/* negative literal -> fix to one */
				if (domain.lb(var) < 0.5)
				{
					if (domain.ub(var) < 0.5)
					{
						// consoleLog("{} = 1 inconsistent with current ub = {}", domain.cNames[var], domain.ub(var));
						infeas = true;
						break;
					}
					infeas = engine.changeLowerBound(var, 1.0);
					FP_ASSERT(!infeas);
				}
			}
		}

		if (infeas)
			break;
	}
}

void CliquesPropagator::propagate(PropagationEngine &engine, Domain::iterator mark, bool initialProp)
{
	const Domain &domain = engine.getDomain();
	int n = domain.ncols();
	std::queue<int> qvars;

	if (initialProp)
	{
		/* Pretend all fixed binaries have yet to be processed */
		for (int var = 0; var < n; var++)
		{
			if ((domain.lb(var) == domain.ub(var)) &&
				(domain.type(var) == 'B'))
			{
				qvars.push(var);
				// if (domain.lb(var) > 0.5)  qvar.push(posLit(var, n));
				// else                       qlits.push(negLit(var, n));
			}
		}
	}
	else
	{
		/* Just look at the bound changes since last propagation */
		Domain::iterator itr = lastPropagated;
		while (itr != mark)
		{
			/* get bound change */
			const auto &bdchg = domain.change(itr++);
			int var = bdchg.var;

			if (domain.type(var) == 'B')
				qvars.push(var);
		}
	}

	/* process queue */
	bool infeas = false;
	while (!qvars.empty() && !infeas)
	{
		int var = qvars.front();
		qvars.pop();
		if (domain.lb(var) > 0.5)
			propagateOneLiteral(engine, posLit(var, n), infeas);
		else if (domain.ub(var) < 0.5)
			propagateOneLiteral(engine, negLit(var, n), infeas);
		else
		{ /* skip non-fixed var */
			;
		}
	}
}

/* Propagate all implications that have x[binvar]=value as premise */
void ImplPropagator::forwardProp(PropagationEngine &engine, int binvar, bool value, bool &infeas)
{
	const Domain &domain = engine.getDomain();
	int n = domain.ncols();
	for (const auto &impl : impltable.getImplsByBin(binvar, value))
	{
		const auto [impliedvar, isUpper] = getImplied(impl.implied, n);
		if (isUpper)
		{
			/* Implied upper bound */
			if (domain.isNewUpperBoundAcceptable(impliedvar, impl.bound))
			{
				infeas = engine.changeUpperBound(impliedvar, impl.bound);
				if (infeas)
					break;
			}
		}
		else
		{
			/* Implied lower bound */
			if (domain.isNewLowerBoundAcceptable(impliedvar, impl.bound))
			{
				infeas = engine.changeLowerBound(impliedvar, impl.bound);
				if (infeas)
					break;
			}
		}
	}
}

/* Backward propagate implications that impose a bound on impliedvar
 *
 * The logic is a follows. Let's say we have a current upper bound on impliedvar.
 * Then we scan all the implications implying a _lower bound_ on implied var
 * (notice that they are sorted such that l1 >= l2 >= .... >= lk) and for each
 * that is violated we derive the negated premise.
 */
void ImplPropagator::backwardProp(PropagationEngine &engine, int impliedvar, bool isUpper, double bound, bool &infeas)
{
	const Domain &domain = engine.getDomain();
	int n = domain.ncols();
	double mult = isUpper ? +1.0 : -1.0;
	for (const auto &impl : impltable.getImplsByImplied(impliedvar, !isUpper))
	{
		/* Stop if the implied bound is not longer negated */
		if (lessEqualThan(mult * impl.bound, mult * bound, domain.feasTol))
			break;

		const auto [binvar, isPos] = varFromLit(impl.lit, n);
		if (!isPos)
		{
			/* The implication was with binvar=0 -> fix to 1 */
			infeas = engine.changeLowerBound(binvar, 1.0);
			if (infeas)
				break;
		}
		else
		{
			/* The implication was with binvar=1 -> fix to 0 */
			infeas = engine.changeUpperBound(binvar, 0.0);
			if (infeas)
				break;
		}
	}
}

void ImplPropagator::propagate(PropagationEngine &engine, Domain::iterator mark, bool initialProp)
{
	const Domain &domain = engine.getDomain();
	int n = domain.ncols();
	bool infeas = false;
	std::queue<int> qvars;

	if (initialProp)
	{
		/* Try to propagate each and every var */
		for (int var = 0; var < n; var++)
		{
			if (domain.type(var) == 'B')
			{
				/* Forward propagation: propagate fixed binaries */
				if (domain.lb(var) > 0.5)
					forwardProp(engine, var, true, infeas);
				else if (domain.ub(var) < 0.5)
					forwardProp(engine, var, false, infeas);
				if (infeas)
					break;
			}
			else
			{
				/* Backward propagation: try both bounds */
				backwardProp(engine, var, false, domain.lb(var), infeas);
				if (infeas)
					break;
				backwardProp(engine, var, true, domain.ub(var), infeas);
				if (infeas)
					break;
			}
		}
	}
	else
	{
		/* Just look at the bound changes since last propagation */
		Domain::iterator itr = lastPropagated;
		while (itr != mark)
		{
			/* get bound change */
			const auto &bdchg = domain.change(itr++);
			qvars.push(bdchg.var);
		}
	}

	/* process queue */
	while (!qvars.empty() && !infeas)
	{
		int var = qvars.front();
		qvars.pop();

		if (domain.type(var) == 'B')
		{
			/* Forward propagation: propagate fixed binaries */
			if (domain.lb(var) > 0.5)
				forwardProp(engine, var, true, infeas);
			else if (domain.ub(var) < 0.5)
				forwardProp(engine, var, false, infeas);
			if (infeas)
				break;
		}
		else
		{
			/* Backward propagation: try both bounds */
			backwardProp(engine, var, false, domain.lb(var), infeas);
			if (infeas)
				break;
			backwardProp(engine, var, true, domain.ub(var), infeas);
			if (infeas)
				break;
		}
	}
}
