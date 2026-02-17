/**
 * @file   linear_propagator.h
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 */

#include "linear_propagator.h"
#include <consolelog.h>

LinearPropagator::LinearPropagator(const MIPInstance& mip_) : PropagatorI{},
	obj_sense{mip_.objSense == 1.0 ? 'L' : 'G'}, mip{mip_} {
		/* Default cutoff is infinity. */
		obj_rhs = mip.objSense * INFTY;

		/* The objective gets propagated as c'x + offset <= cutoff for minimization (objsense == 1.0), so c'x <= cutoff - offset. */
		FP_ASSERT(mip.objSense == 1.0);
	}

/* Compute activities for a given row */
LinearPropagator::State LinearPropagator::computeState(const Domain &domain, const int row) const
{
	State s{}; //< all fields initialized to zero
	int n = domain.ncols();
	const bool is_objective = (row == mip.nrows);

	const int *idx = is_objective ? mip.obj_cols.data() : mip.rows[row].idx();
	const double *coefs = is_objective ? mip.obj_coefs.data() : mip.rows[row].coef();
	const size_t nnz = is_objective ? mip.obj_coefs.size() : mip.rows[row].size();

	for (size_t inz = 0; inz < nnz; ++inz)
	{
		const int var = idx[inz];
		const double coef = coefs[inz];

		FP_ASSERT((var >= 0) && (var < n));
		FP_ASSERT(coef != 0.0);
		double lb = domain.lb(var);
		FP_ASSERT(lb > -domain.infinity);
		double ub = domain.ub(var);
		FP_ASSERT(ub < +domain.infinity);

		/* update diameter */
		double spread = ub - lb;
		if (domain.type(var) == 'C')
			spread *= (1.0 - domain.minContRed);
		else
			spread = std::max(spread - domain.feasTol, 0.0);
		s.diameter = std::max(s.diameter, fabs(coef) * spread);
	}

	return s;
}

void LinearPropagator::update_obj_cutoff(double obj_cutoff) {
	obj_rhs = obj_cutoff - mip.objOffset;
}

void LinearPropagator::init(const Domain &domain)
{
	int m = mip.nrows + 1; /* All rows + objective. Objective row [nrows]. */
	states.resize(m);
	for (int i = 0; i < m; i++)
		states[i] = computeState(domain, i);

	/* Rows are normalized: find position of first non binary variable.
	 * This is not part of computeState as this piece of information nevers gets invalidated.
	 */
	firstNonBin.resize(m);
	for (int i = 0; i < m; i++)
	{
		const bool is_objective = (i == mip.nrows);
		const int *idx = is_objective ? mip.obj_cols.data() : mip.rows[i].idx();
		const size_t cnt = is_objective ? mip.obj_cols.size() : mip.rows[i].size();
		size_t k = 0;

		while ((k < cnt) && domain.type(idx[k]) == 'B')
			k++;
		firstNonBin[i] = k;
	}

	lastUpdated = domain.changesEnd();
}

void LinearPropagator::update(const Domain &domain, Domain::iterator mark)
{
	FP_ASSERT(lastUpdated <= mark);

	while (lastUpdated < mark)
	{
		const BoundChange &bdchg = domain.change(lastUpdated++);

		/* Shifts reset the infeasibility status! */
		if (bdchg.type != BoundChange::Type::SHIFT)
			continue;

		for (const auto &[i, coef] : mip.cols[bdchg.var])
			states[i].infeas = false;

		if (!iszero(mip.obj[bdchg.var]))
			states[mip.nrows].infeas = false;
	}
}

void LinearPropagator::undoBoundChangeForEntry(const Domain &domain, const BoundChange &bdchg, int row, double coef) {
	State &s = states[row];
	const int col = bdchg.var;

	/* Reset infeas status on backtrack */
	s.infeas = false;

	/* update diameter */
	if (bdchg.type != BoundChange::Type::SHIFT)
	{
		double origLB = mip.lb[col];
		double origUB = mip.ub[col];
		FP_ASSERT(origUB >= origLB);
		double spread = origUB - origLB;
		if (domain.type(col) == 'C')
			spread *= (1.0 - domain.minContRed);
		else
			spread = std::max(spread - domain.feasTol, 0.0);
		s.diameter = std::max(s.diameter, fabs(coef) * spread);
	}
}

/* Reverse activity update by undoing a given bound change */
void LinearPropagator::undoBoundChange(const Domain &domain, const BoundChange &bdchg)
{
	const int col = bdchg.var;

	for (const auto &[row, coef] : mip.cols[col])
		undoBoundChangeForEntry(domain, bdchg, row, coef);

	const double obj_coef = mip.obj[col];
	if (!iszero(obj_coef))
		undoBoundChangeForEntry(domain, bdchg, mip.nrows, obj_coef);
}

void LinearPropagator::undo(const Domain &domain, Domain::iterator mark)
{
	/* Nothing to do if we are backtracking to a position that is still
	 * in the future w.r.t. this propagator.
	 */
	if (lastUpdated <= mark)
		return;

	/* update activities incrementally */
	while (lastUpdated != mark)
		undoBoundChange(domain, domain.change(--lastUpdated));
}

/* Propagate row i as a <= constraint */
void LinearPropagator::propagateOneRowLessThan(PropagationEngine &engine, int i, double mult, double bound)
{
	const bool is_objective = (i == mip.nrows);
	State &s = states[i];

	FP_ASSERT((mult == 1.0) || (mult == -1.0));

	const Domain &domain = engine.getDomain();

	/* Perform simple checks first */
	double act = (mult > 0) ? engine.getMinAct(i) : engine.getMaxAct(i);
	double slack = mult * (bound - act);

	if (slack < -domain.feasTol)
	{
		// consoleLog("Row {} mult={} infeasible", mip.rNames[i], mult);
		s.infeas = true;
		return;
	}

	/* Make sure slack is non-negative */
	if (slack < 0.0)
		slack = 0.0;

	if (lessEqualThan(s.diameter, slack, domain.feasTol))
	{
		// consoleLog("Row {} mult={} cannot propagate: {} <= {}", mip.rNames[i], mult, s.diameter, slack);
		// consoleLog("minAct={} maxAct={}", s.minAct, s.maxAct);
		return;
	}

	// consoleLog("Tighten vars from row {} mult={}", mip.rNames[i], mult);

	/* Loop over the row and tighten variables */
	bool infeas = false;
	const int *idx = is_objective ? mip.obj_cols.data() : mip.rows[i].idx();
	const double *coefs = is_objective ? mip.obj_coefs.data() : mip.rows[i].coef();
	const size_t nnz = is_objective ? mip.obj_cols.size() : mip.rows[i].size();

	/* First loop over binaries */
	for (int k = 0; k < firstNonBin[i]; k++)
	{
		int j = idx[k];
		double coef = mult * coefs[k];

		/* Stop if abs(coef) is below slack (rows are normalized!) */
		if (lessEqualThan(fabs(coef), slack, domain.feasTol))
			break;

		/* Skip fixed binaries */
		if (domain.lb(j) == domain.ub(j))
			continue;

		FP_ASSERT(fabs(coef) > slack);
		if (coef > 0.0)
		{
			/* Fix binary to zero */
			infeas = engine.changeUpperBound(j, 0.0);
			FP_ASSERT(!infeas);
		}
		else
		{
			/* Fix binary to one */
			infeas = engine.changeLowerBound(j, 1.0);
			FP_ASSERT(!infeas);
		}
	}
	/* Then the rest */
	for (size_t k = firstNonBin[i]; k < nnz; k++)
	{
		int j = idx[k];
		double coef = mult * coefs[k];

		/* Do not derive bounds from tiny coefficients */
		if (fabs(coef) <= domain.feasTol)
			continue;

		/* Skip fixed vars */
		if ((domain.ub(j) - domain.lb(j)) <= domain.feasTol)
			continue;

		if (coef > 0.0)
		{
			double delta = slack / coef;
			FP_ASSERT(delta >= 0.0);
			double newBound = domain.lb(j) + delta;
			if (domain.type(j) == 'I')
				newBound = floorEps(newBound, domain.feasTol);
			if (domain.isNewUpperBoundAcceptable(j, newBound))
			{
				infeas = engine.changeUpperBound(j, newBound);
				FP_ASSERT(!infeas);
			}
		}
		else if (coef < 0.0)
		{
			double delta = slack / coef;
			FP_ASSERT(delta <= 0.0);
			double newBound = domain.ub(j) + delta;
			if (domain.type(j) == 'I')
				newBound = ceilEps(newBound, domain.feasTol);
			if (domain.isNewLowerBoundAcceptable(j, newBound))
			{
				infeas = engine.changeLowerBound(j, newBound);
				FP_ASSERT(!infeas);
			}
		}
	}

	update(domain, domain.changesEnd());
}

/* propagate a single linear constraint */
void LinearPropagator::propagateOneRow(PropagationEngine &engine, int i)
{
	const bool is_objective = (i == mip.nrows);
	const char sense = is_objective ? obj_sense : mip.sense[i];
	const double rhs = is_objective ? obj_rhs : mip.rhs[i];
	if (is_objective && obj_rhs == INFTY)
		return;

	bool rowHasLB = (sense != 'L');
	bool rowHasUB = (sense != 'G');

	const Domain &domain = engine.getDomain();
	State &s = states[i];
	if (s.infeas)
		return;

	/* Is the whole row fixed? */
	double minAct = engine.getMinAct(i);
	double maxAct = engine.getMaxAct(i);
	if ((fabs(maxAct - minAct) <= domain.feasTol))
	{
		/* Check if the row became infeasible */
		if (rowHasLB && lessThan(maxAct, rhs, domain.feasTol))
		{
			// consoleLog("Row {} mult={} infeasible", mip.rNames[i], 1.0);
			states[i].infeas = true;
		}
		else if (rowHasUB && greaterThan(minAct, rhs, domain.feasTol))
		{
			// consoleLog("Row {} mult={} infeasible", mip.rNames[i], -1.0);
			states[i].infeas = true;
		}
		/* If not, then all variables must be fixed and there is nothing to do anyway */
		return;
	}

	if (rowHasUB)
		propagateOneRowLessThan(engine, i, 1.0, rhs);
	if (rowHasLB)
		propagateOneRowLessThan(engine, i, -1.0, rhs);
}

void LinearPropagator::propagate(PropagationEngine &engine, Domain::iterator mark, bool initialProp)
{
	/* Update state */
	const Domain &domain = engine.getDomain();
	update(domain, domain.changesEnd());

	std::queue<int> qrows;
	if (initialProp)
	{
		/* Enqueue all rows */
		for (int i = 0; i < mip.nrows; i++)
			qrows.push(i);

		qrows.push(mip.nrows); // The objective
	}
	else
	{
		/* Loop over list of bound changes */
		FP_ASSERT(lastPropagated <= mark);
		Domain::iterator itr = lastPropagated;
		while (itr != mark)
		{
			const auto &bdchg = domain.change(itr++);
			int j = bdchg.var;
			for (const auto &[i, coef] : mip.cols[j])
			{
				const State &s = states[i];

				if (s.infeas)
					continue;

				qrows.push(i);
			}

			/* Potentially propagate the objective. */
			const double obj_coef = mip.obj[j];
			if (!iszero(obj_coef)) {
				const State &s = states[mip.nrows];
				if (!s.infeas) {
					qrows.push(mip.nrows);
				}
			}
		}
	}

	while (!qrows.empty())
	{
		int i = qrows.front();
		qrows.pop();
		propagateOneRow(engine, i);
		if (states[i].infeas)
		{
			// consoleLog("Linear propagator infeasibility on row {}", i);
			break;
		}
	}
}

void LinearPropagator::commit(const Domain &domain)
{
	Domain::iterator end = domain.changesEnd();
	update(domain, end);
	lastUpdated = domain.changesBegin();
}
