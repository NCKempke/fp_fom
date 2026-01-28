/**
 * @file   linear_propagator.h
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 */

#include "linear_propagator.h"
#include <consolelog.h>

LinearPropagator::LinearPropagator(const MIPInstance& mip_) : PropagatorI{}, mip{mip_} {}

/* Compute activities for a given row */
LinearPropagator::State LinearPropagator::computeState(const Domain &domain, SparseMatrix::view_type row) const
{
	State s{}; //< all fields initialized to zero
	int n = domain.ncols();

	for (const auto &[var, coef] : row)
	{
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

void LinearPropagator::init(const Domain &domain)
{
	int m = mip.nrows;
	states.resize(m);
	for (int i = 0; i < m; i++)
		states[i] = computeState(domain, mip.rows[i]);

	/* Rows are normalized: find position of first non binary variable.
	 * This is not part of computeState as this piece of information nevers gets invalidated.
	 */
	firstNonBin.resize(m);
	for (int i = 0; i < m; i++)
	{
		const auto &row = mip.rows[i];
		const int *idx = row.idx();
		int cnt = row.size();
		int k = 0;

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
	}
}

/* Reverse activity update by undoing a given bound change */
void LinearPropagator::undoBoundChange(const Domain &domain, const BoundChange &bdchg)
{
	int j = bdchg.var;

	for (const auto &[i, coef] : mip.cols[j])
	{
		State &s = states[i];

		/* Reset infeas status on backtrack */
		s.infeas = false;

		/* update diameter */
		if (bdchg.type != BoundChange::Type::SHIFT)
		{
			double origLB = mip.lb[j];
			double origUB = mip.ub[j];
			FP_ASSERT(origUB >= origLB);
			double spread = origUB - origLB;
			if (domain.type(j) == 'C')
				spread *= (1.0 - domain.minContRed);
			else
				spread = std::max(spread - domain.feasTol, 0.0);
			s.diameter = std::max(s.diameter, fabs(coef) * spread);
		}
	}
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
	const auto &row = mip.rows[i];
	const int *idx = row.idx();
	const double *coefs = row.coef();
	int size = row.size();

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
	for (int k = firstNonBin[i]; k < size; k++)
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
	char sense = mip.sense[i];
	double rhs = mip.rhs[i];

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
		}
	}

	while (!qrows.empty())
	{
		int i = qrows.front();
		qrows.pop();
		propagateOneRow(engine, i);
		if (states[i].infeas)
		{
			// consoleLog("Linear propagator infeasiblity on row {}", i);
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
