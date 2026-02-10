/**
 * @file   propagation.c
 * @brief  Constraint Propagation API
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 */

#include "propagation.h"

#include <tool_assert.h>
#include <floats.h>
#include <consolelog.h>

void Domain::init(std::span<const double> _lb, std::span<const double> _ub, std::span<const char> _xtype)
{
	FP_ASSERT(_lb.size() == _ub.size());
	FP_ASSERT(_ub.size() == _xtype.size());
	xlb.resize(_lb.size());
	xub.resize(_ub.size());
	xtype.resize(_xtype.size());
	std::copy(_lb.begin(), _lb.end(), xlb.begin());
	std::copy(_ub.begin(), _ub.end(), xub.begin());
	std::copy(_xtype.begin(), _xtype.end(), xtype.begin());
	stack.clear();
	for (int j = 0; j < ncols(); j++)
	{
		/* Make sure that all binaries have proper bounds */
		FP_ASSERT(lb(j) <= ub(j));
		if (type(j) == 'B')
		{
			FP_ASSERT((lb(j) == 0.0) || (lb(j) == 1.0));
			FP_ASSERT((ub(j) == 0.0) || (ub(j) == 1.0));
		}
		/* Heuristically clip infinite bounds to some (not too large) value */
		if (xlb[j] <= -infinity)
			xlb[j] = -infinity / 1000.0;
		if (xub[j] >= +infinity)
			xub[j] = +infinity / 1000.0;
	}
}

bool Domain::isNewLowerBoundAcceptable(int var, double newBound) const
{
	double currLB = lb(var);
	double currUB = ub(var);
	FP_ASSERT(currLB <= currUB);

	if (fabs(newBound) >= infinity)
		return false;

	double delta = newBound - currLB;
	if (delta <= feasTol)
		return false;

	if (type(var) == 'C')
	{
		double thr = minContRed * (currUB - currLB);
		if (delta <= thr)
			return false;
	}

	// consoleLog("Accepting new LB {} for {}; delta {}", newBound, var, delta);
	return true;
}

bool Domain::isNewUpperBoundAcceptable(int var, double newBound) const
{
	double currLB = lb(var);
	double currUB = ub(var);
	FP_ASSERT(currLB <= currUB);

	if (fabs(newBound) >= infinity)
		return false;

	double delta = currUB - newBound;
	if (delta <= feasTol)
		return false;

	if (type(var) == 'C')
	{
		double thr = minContRed * (currUB - currLB);
		if (delta <= thr)
			return false;
	}

	// consoleLog("Accepting new UB {} for {}; delta {}", newBound, var, delta);
	return true;
}

bool Domain::changeLowerBound(int var, double newBound)
{
	int n = ncols();
	FP_ASSERT((var >= 0) && (var < n));

	/* collect the current bounds of the column */
	double oldLowerBound = lb(var);
	double oldUpperBound = ub(var);
	FP_ASSERT(oldLowerBound <= oldUpperBound + zeroTol);
	FP_ASSERT(oldLowerBound <= newBound + zeroTol);

	/* detect infeasibilities */
	if (greaterThan(newBound, oldUpperBound, feasTol))
	{
		// consoleLog("{} >= {} inconsistent with ub = {}", cNames[var], newBound, oldUpperBound);
		return true;
	}
	else if (greaterEqualThan(newBound, oldUpperBound, zeroTol))
	{
		/* if the bounds are nearly identical, make them the same */
		newBound = oldUpperBound;
	}

	/* nothing to do if nothing changed */
	if (equal(newBound, oldLowerBound, zeroTol))
		return false;

	// consoleLog("{} >= {}", cNames[var], newBound);

	/* push bound change to the stack */
	stack.emplace_back(var, BoundChange::Type::LOWER, newBound, oldLowerBound);

	/* actually modify bound */
	xlb[var] = newBound;
	return false;
}

bool Domain::changeUpperBound(int var, double newBound)
{
	int n = ncols();
	FP_ASSERT((var >= 0) && (var < n));

	/* collect the current bounds of the column */
	double oldLowerBound = lb(var);
	double oldUpperBound = ub(var);
	FP_ASSERT(oldLowerBound <= oldUpperBound + zeroTol);
	FP_ASSERT(newBound <= oldUpperBound + zeroTol);

	/* detect infeasibilities */
	if (lessThan(newBound, oldLowerBound, feasTol))
	{
		// consoleLog("{} <= {} inconsistent with lb = {}", cNames[var], newBound, oldLowerBound);
		return true;
	}
	else if (lessEqualThan(newBound, oldLowerBound, zeroTol))
	{
		/* if the bounds are nearly identical, make them the same */
		newBound = oldLowerBound;
	}

	/* nothing to do if nothing changed */
	if (equal(newBound, oldUpperBound, zeroTol))
		return false;

	// consoleLog("{} <= {}", cNames[var], newBound);

	/* push bound change to the stack */
	stack.emplace_back(var, BoundChange::Type::UPPER, newBound, oldUpperBound);

	/* actually modify bound */
	xub[var] = newBound;
	return false;
}

void Domain::shift(int var, double newValue)
{
	int n = ncols();
	FP_ASSERT((var >= 0) && (var < n));

	/* Can only shift fixed variables */
	double oldLowerBound = lb(var);
	double oldUpperBound = ub(var);
	FP_ASSERT(oldLowerBound == oldUpperBound);

	/* nothing to do if nothing changed */
	if (newValue == oldUpperBound)
		return;

	/* push bound change to the stack */
	stack.emplace_back(var, BoundChange::Type::SHIFT, newValue, oldUpperBound);

	/* actually modify bounds */
	xlb[var] = newValue;
	xub[var] = newValue;
}

void Domain::undoLast()
{
	const BoundChange &bdchg = lastChange();
	switch (bdchg.type)
	{
	case BoundChange::Type::SHIFT:
		xlb[bdchg.var] = bdchg.oldValue;
		xub[bdchg.var] = bdchg.oldValue;
		break;
	case BoundChange::Type::UPPER:
		xub[bdchg.var] = bdchg.oldValue;
		break;
	case BoundChange::Type::LOWER:
		xlb[bdchg.var] = bdchg.oldValue;
		break;
	default:
		FP_ASSERT(false);
	}

	/* resize stack */
	stack.pop_back();
}

void Domain::undo(Domain::iterator mark)
{
	/* undo bound changes: we must do so in reverse order! */
	Domain::iterator itr = changesEnd();
	FP_ASSERT(itr >= mark);
	size_t dist = itr - mark;
	for (size_t i = 0; i < dist; i++)
		undoLast();
	FP_ASSERT(changesEnd() == mark);
}

PropagationEngine::PropagationEngine(const MIPInstance& mip_) : mip(mip_), obj_sense{mip_.objSense == 1.0 ? 'L' : 'G'},
															 minAct(mip.nrows + 1),
															 maxAct(mip.nrows + 1),
															 violated(mip.nrows + 1) {}

PropagatorPtr PropagationEngine::getPropagator(const std::string &name) const
{
	for (PropagatorPtr prop : propagators)
	{
		if (prop->name() == name)
			return prop;
	}
	return PropagatorPtr{};
}

static inline bool needsRecomputation(double oldValue, double newValue)
{
	return (fabs(newValue) < (fabs(oldValue) * 1e-3));
}

void PropagationEngine::update_obj_cutoff(double obj_cutoff) {
	obj_rhs = obj_cutoff - mip.objOffset;

	/* Recompute the objective row violation. */
	recomputeViolationRow(mip.nrows);

	for (auto& prop : propagators)
		prop->update_obj_cutoff(obj_cutoff);
}

void PropagationEngine::init(std::span<const double> _lb, std::span<const double> _ub, std::span<const char> _xtype)
{
	/* Default cutoff is infinity. */
	obj_rhs = mip.objSense * INFTY;

	/* Initialize domain */
	FP_ASSERT(_lb.size() == _ub.size());
	FP_ASSERT(_ub.size() == _xtype.size());
	domain.init(_lb, _ub, _xtype);
	domain.cNames = mip.cNames;

	/* Evaluate current minimum and maximum activities for each row */
	int m = mip.nrows + 1;
	for (int i = 0; i < m; i++)
	{
		recomputeRowActivity(i);
	}

	/* Compute current violation and set of violated constraints */
	recomputeViolation();

	/* Initialize propagators */
	for (const auto &prop : propagators)
		prop->init(domain);
}

double PropagationEngine::changeLowerBoundRow(int row, double coef, double delta) {
	const bool is_objective = (row == mip.nrows);
	const char sense = is_objective ? obj_sense : mip.sense[row];
	const double rhs = is_objective ? obj_rhs : mip.rhs[row];

	double oldViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	bool recomp = false;
	if (coef > 0.0)
	{
		double oldAct = minAct[row];
		minAct[row] += (coef * delta);
		recomp |= needsRecomputation(oldAct, minAct[row]);
	}
	else
	{
		double oldAct = maxAct[row];
		maxAct[row] += (coef * delta);
		recomp |= needsRecomputation(oldAct, maxAct[row]);
	}
	if (recomp)
		recomputeRowActivity(row);
	double newViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	if (newViol > domain.feasTol)
		violated.add(row);
	else
		violated.remove(row);

#ifdef DEBUG_EXPENSIVE
	debugCheckRow(i);
#endif
	return newViol - oldViol;
}

bool PropagationEngine::changeLowerBound(int var, double newBound)
{
	double oldBound = domain.lb(var);
	bool infeas = domain.changeLowerBound(var, newBound);
	if (infeas)
		return infeas;

	newBound = domain.lb(var);
	if (newBound == oldBound)
		return infeas;

	double delta = newBound - oldBound;

	double deltaViol = 0.0;

	for (const auto &[row, coef] : mip.cols[var])
		deltaViol += changeLowerBoundRow(row, coef, delta);

	const double obj_coef = mip.obj[var];
	if (!iszero(obj_coef))
		deltaViol += changeLowerBoundRow(mip.nrows, obj_coef, delta);

	double oldTotViol = totViol;
	totViol += deltaViol;
	if (needsRecomputation(oldTotViol, totViol))
		recomputeViolation();

	return infeas;
}

double PropagationEngine::changeUpperBoundRow(int row, double coef, double delta) {
	const bool is_objective = (row == mip.nrows);
	const char sense = is_objective ? obj_sense : mip.sense[row];
	const double rhs = is_objective ? obj_rhs : mip.rhs[row];

	double oldViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	bool recomp = false;
	if (coef < 0.0)
	{
		double oldAct = minAct[row];
		minAct[row] += (coef * delta);
		recomp |= needsRecomputation(oldAct, minAct[row]);
	}
	else
	{
		double oldAct = maxAct[row];
		maxAct[row] += (coef * delta);
		recomp |= needsRecomputation(oldAct, maxAct[row]);
	}
	if (recomp)
		recomputeRowActivity(row);

	double newViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	if (newViol > domain.feasTol)
		violated.add(row);
	else
		violated.remove(row);

#ifdef DEBUG_EXPENSIVE
	debugCheckRow(i);
#endif
	return newViol - oldViol;
}

bool PropagationEngine::changeUpperBound(int var, double newBound)
{
	double oldBound = domain.ub(var);
	bool infeas = domain.changeUpperBound(var, newBound);
	if (infeas)
		return infeas;

	newBound = domain.ub(var);
	if (newBound == oldBound)
		return infeas;

	double delta = newBound - oldBound;

	double deltaViol = 0.0;
	for (const auto &[row, coef] : mip.cols[var])
		deltaViol += changeUpperBoundRow(row, coef, delta);

	const double obj_coef = mip.obj[var];
	if (!iszero(obj_coef))
		deltaViol += changeUpperBoundRow(mip.nrows, obj_coef, delta);

	double oldTotViol = totViol;
	totViol += deltaViol;
	if (needsRecomputation(oldTotViol, totViol))
		recomputeViolation();

	return infeas;
}

bool PropagationEngine::fix(int var, double value)
{
	bool infeas = changeLowerBound(var, value);
	if (infeas)
		return infeas;
	infeas = changeUpperBound(var, value);
	return infeas;
}

double PropagationEngine::shift_row(int row, double coef, double delta) {
	const bool is_objective = (row == mip.nrows);
	const char sense = is_objective ? obj_sense : mip.sense[row];
	const double rhs = is_objective ? obj_rhs : mip.rhs[row];

	double oldViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	bool recomp = false;

	double oldMinAct = minAct[row];
	minAct[row] += (coef * delta);
	recomp |= needsRecomputation(oldMinAct, minAct[row]);

	double oldMaxAct = maxAct[row];
	maxAct[row] += (coef * delta);
	recomp |= needsRecomputation(oldMaxAct, maxAct[row]);
	if (recomp)
		recomputeRowActivity(row);
	double newViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	if (newViol > domain.feasTol)
		violated.add(row);
	else
		violated.remove(row);

#ifdef DEBUG_EXPENSIVE
	debugCheckRow(row);
#endif
	return newViol - oldViol;
}

void PropagationEngine::shift(int var, double newValue)
{
	FP_ASSERT(domain.lb(var) == domain.ub(var));
	double oldValue = domain.lb(var);
	domain.shift(var, newValue);

	newValue = domain.lb(var);
	if (newValue == oldValue)
		return;

	double delta = newValue - oldValue;

	double deltaViol = 0.0;
	for (const auto &[row, coef] : mip.cols[var])
		deltaViol += shift_row(row, coef, delta);

	const double obj_coef = mip.obj[var];
	if (!iszero(obj_coef))
		deltaViol += shift_row(mip.nrows, obj_coef, delta);

	double oldTotViol = totViol;
	totViol += deltaViol;
	if (needsRecomputation(oldTotViol, totViol))
		recomputeViolation();
}

void PropagationEngine::recomputeRowActivity(int i)
{
	computeActivity(i, minAct[i], maxAct[i]);
}

bool PropagationEngine::propagate(bool initialProp)
{
	/* Nothing to do if we are already infeasible */
	if (!violated.empty())
		return true;

	int nPasses = 0;
	bool infeas = false;

	while (!infeas)
	{
		// consoleLog("Propagation pass {}", nPasses);
		/* stop if we are taking too many passes */
		if (nPasses >= maxPasses)
			break;

		Domain::iterator prevEnd = domain.changesEnd();

		/* One pass through all propagators */
		for (const auto &prop : propagators)
		{
			/* Skip disable propagators */
			if (!prop->enabled())
				continue;

			/* Run each propagator until it reaches its own fixpoint or infeasibility */
			bool initialFirstPass = initialProp;

			while (true)
			{
				Domain::iterator currEnd = domain.changesEnd();
				FP_ASSERT(currEnd >= prop->lastPropagated);
				bool hasChanges = (prop->lastPropagated != currEnd);

				/* Fixpoint reached */
				if (!hasChanges && !initialFirstPass)
					break;

				/* Propagate */
				prop->propagate(*this, currEnd, initialFirstPass);
				prop->lastPropagated = currEnd;
				initialFirstPass = false;

				/* Stop on infeasibility */
				if (!violated.empty())
				{
					// consoleLog("{} infeasible", prop->name());
					infeas = true;
					break;
				}
			}

			/* No need to run the next propagators */
			if (infeas)
				break;
		}

		/* Stop if there was no additional reduction in the last pass */
		if (domain.changesEnd() == prevEnd)
			break;

		/* Stop if infeasibility was detected */
		if (infeas)
			break;

		nPasses++;
	}

	return infeas;
}

bool PropagationEngine::directImplications()
{
	Domain::iterator currEnd = domain.changesEnd();

	/* One pass through all propagators */
	for (const auto &prop : propagators)
	{
		/* Skip disable propagators */
		if (!prop->enabled())
			continue;

		/* Run each propagator once */
		FP_ASSERT(currEnd >= prop->lastPropagated);

		/* Propagate */
		prop->propagate(*this, currEnd, false);
		prop->lastPropagated = currEnd;

		/* Stop on infeasibility */
		if (!violated.empty())
		{
			// consoleLog("{} infeasible", prop->name());
			return true;
		}
	}

	return false;
}

double PropagationEngine::undo_row(const BoundChange& bdchg, int row, double coef, double delta) {
	const bool is_objective = (row == mip.nrows);
	const char sense = is_objective ? obj_sense : mip.sense[row];
	const double rhs = is_objective ? obj_rhs : mip.rhs[row];

	double oldViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	double oldMinAct = minAct[row];
	double oldMaxAct = maxAct[row];
	switch (bdchg.type)
	{
	case BoundChange::Type::SHIFT:
		minAct[row] -= (coef * delta);
		maxAct[row] -= (coef * delta);
		break;
	case BoundChange::Type::UPPER:
		if (coef > 0.0)
			maxAct[row] -= (coef * delta);
		else
			minAct[row] -= (coef * delta);
		break;
	case BoundChange::Type::LOWER:
		if (coef > 0.0)
			minAct[row] -= (coef * delta);
		else
			maxAct[row] -= (coef * delta);
		break;
	default:
		FP_ASSERT(false);
	}
	if (needsRecomputation(oldMinAct, minAct[row]) ||
		needsRecomputation(oldMaxAct, maxAct[row]))
	{
		recomputeRowActivity(row);
	}
	double newViol = rowViol(minAct[row], maxAct[row], sense, rhs);
	if (newViol > domain.feasTol)
		violated.add(row);
	else
		violated.remove(row);
#ifdef DEBUG_EXPENSIVE
	debugCheckRow(row);
#endif

	return (newViol - oldViol);
}

void PropagationEngine::undo(Domain::iterator mark)
{
	/* Backtrack propagators' states */
	for (const auto &prop : propagators)
	{
		prop->undo(domain, mark);
		if (prop->lastPropagated > mark)
		{
			prop->lastPropagated = mark;
		}
	}

	/* Backtrack domain one change at time, updating activites */
	Domain::iterator itr = domain.changesEnd();
	FP_ASSERT(itr >= mark);
	size_t dist = itr - mark;
	for (size_t i = 0; i < dist; i++)
	{
		BoundChange bdchg = domain.lastChange();
		domain.undoLast();
		double delta = bdchg.value - bdchg.oldValue;
		int var = bdchg.var;

		double deltaViol = 0.0;
		for (const auto &[row, coef] : mip.cols[var])
			deltaViol += undo_row(bdchg, row, coef, delta);

		const double obj_coef = mip.obj[var];
		if (!iszero(obj_coef))
			deltaViol += undo_row(bdchg, mip.nrows, obj_coef, delta);

		totViol += deltaViol;
	}
	FP_ASSERT(domain.changesEnd() == mark);
}

void PropagationEngine::commit()
{
	for (const auto &prop : propagators)
	{
		prop->commit(domain);
		prop->lastPropagated = domain.changesBegin();
	}

	domain.commit();
}

void PropagationEngine::enableAll()
{
	for (PropagatorPtr prop : propagators)
		prop->enable();
}

void PropagationEngine::disableAll()
{
	for (PropagatorPtr prop : propagators)
		prop->disable();
}

double PropagationEngine::recomputeViolationRow(int row) {
	const bool is_objective = (row == mip.nrows);
	const double rhs = is_objective ? obj_rhs : mip.rhs[row];
	const char sense = is_objective ? obj_sense : mip.sense[row];

	double viol = rowViol(minAct[row], maxAct[row], sense, rhs);
	FP_ASSERT(viol >= 0.0);

	if (viol > domain.feasTol)
	{
		// consoleLog("Violation detected at i = {}", i);
		violated.add(row);
	}

	return viol;
}

/* Compute current violation and set of violated constraints */
void PropagationEngine::recomputeViolation()
{
	int m = mip.nrows + 1;
	violated.clear();
	totViol = 0.0;
	for (int row = 0; row < m; ++row)
		totViol += recomputeViolationRow(row);
	// consoleLog("Total violation = {}", totViol);
}

void PropagationEngine::computeActivity(int row, double &minA, double &maxA) const
{
	const bool is_objective = (row == mip.nrows);
	minA = 0.0;
	maxA = 0.0;

	const int *idx = is_objective ? mip.obj_cols.data() : mip.rows[row].idx();
	const double *coefs = is_objective ? mip.obj_coefs.data() : mip.rows[row].coef();
	const size_t nnz = is_objective ? mip.obj_cols.size() : mip.rows[row].size();

	for (size_t inz = 0; inz < nnz; ++inz)
	{
		const int var = idx[inz];
		const double coef = coefs[inz];

		FP_ASSERT((var >= 0) && (var < mip.ncols));
		FP_ASSERT(coef != 0.0);
		double lb = domain.lb(var);
		FP_ASSERT(lb > -domain.infinity);
		double ub = domain.ub(var);
		FP_ASSERT(ub < +domain.infinity);

		if (coef > 0.0)
		{
			minA += (coef * lb);
			maxA += (coef * ub);
		}
		else
		{
			minA += (coef * ub);
			maxA += (coef * lb);
		}
	}
}

void PropagationEngine::debugCheckRow(int row) const
{
	const bool is_objective = (row == mip.nrows);
	double minA;
	double maxA;
	computeActivity(row, minA, maxA);
	FP_ASSERT(relEqual(minA, minAct[row], domain.zeroTol));
	FP_ASSERT(relEqual(maxA, maxAct[row], domain.zeroTol));

	const char sense = is_objective ? obj_sense : mip.sense[row];
	const double rhs = is_objective ? obj_rhs : mip.rhs[row];

	double viol = rowViol(minAct[row], maxAct[row], sense, rhs);
	FP_ASSERT((viol > domain.feasTol) == violated.has(row));
}

void PropagationEngine::debugChecks() const
{
	int m = mip.nrows + 1;
	for (int i = 0; i < m; i++)
		debugCheckRow(i);
}

void printChangesSinceMark(const Domain &domain, Domain::iterator mark)
{
	Domain::iterator end = domain.changesEnd();
	while (mark != end)
	{
		const BoundChange &bdchg = domain.change(mark++);
		int var = bdchg.var;
		FP_ASSERT((0 <= var) && (var < domain.ncols()));
		switch (bdchg.type)
		{
		case (BoundChange::Type::SHIFT):
			consoleLog("[{}] {} {}->{}", var, domain.cNames[var], bdchg.oldValue, bdchg.value);
			break;
		case (BoundChange::Type::UPPER):
			consoleLog("[{}] {} <= {}", var, domain.cNames[var], bdchg.value);
			break;
		case (BoundChange::Type::LOWER):
			consoleLog("[{}] {} >= {}", var, domain.cNames[var], bdchg.value);
			break;
		default:
			FP_ASSERT(false);
			break;
		}
	}
	consoleLog("");
}
