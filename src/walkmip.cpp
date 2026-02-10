/**
 * @file walkmip.cpp
 * @brief Walkmip repair implementation
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 * @contributor Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2020-2025
 *
 * Copyright 2020 Domenico Salvagnin
 * Copyright 2025 Nils-Christian Kempke
 */

// #define DEBUG_LOG

#ifdef DEBUG_LOG
static int DEBUG_LEVEL = 2;
#endif //< DEBUG_LOG

#include "walkmip.h"
#include "index_queue.h"
#include "tool_app.h"

#include <floats.h>
#include <consolelog.h>
#include <timer.h>

WalkMIP::WalkMIP(const MIPInstance &mip_, const Params &params, PropagationEngine &_engine)
	: mip(mip_), seed(params.seed), randomWalkProbability(params.randomWalkProbability), maxRepairSteps(params.maxRepairSteps), maxRepairNonImprove(params.maxRepairNonImprove), timeLimit(params.timeLimit), repair_objective(params.repair_objective), engine(_engine)
{
	// initialize random engine
	rndgen.seed(seed);
}

void WalkMIP::applyFlip(int var)
{
	const Domain &domain = engine.getDomain();
	FP_ASSERT(domain.type(var) == 'B');
	FP_ASSERT(domain.lb(var) == domain.ub(var));
	double newValue = (domain.lb(var) > 0.5) ? 0.0 : 1.0;
	engine.shift(var, newValue);
	consoleDebug(3, "Apply flip {} = {}: viol = {}", mip.cNames[var], newValue, engine.violation());
}

void WalkMIP::applyShift(int var, double delta)
{
	const Domain &domain = engine.getDomain();
	FP_ASSERT(domain.type(var) == 'I');
	FP_ASSERT(domain.lb(var) == domain.ub(var));
	double newValue = domain.lb(var) + delta;
	engine.shift(var, newValue);
	consoleDebug(3, "Apply shift of {} to {}: viol = {}", delta, mip.cNames[var], engine.violation());
}

void WalkMIP::evalFlipRow(int row, double coef, double deltaX, int pickedRow, bool& isCand, double& damage) {
	const bool is_objective = (row == mip.nrows);
	const double rhs = is_objective ? engine.get_obj_rhs() : mip.rhs[row];
	const char sense = is_objective ? engine.get_obj_sense() : mip.sense[row];

	double minAct = engine.getMinAct(row);
	double maxAct = engine.getMaxAct(row);
	double deltaAct = deltaX * coef;

	double oldViol = rowViol(minAct, maxAct, sense, rhs);
	double newViol = rowViol(minAct + deltaAct, maxAct + deltaAct, sense, rhs);
	if (newViol > oldViol)
		damage += (newViol - oldViol);

	if ((row == pickedRow) && (newViol > oldViol))
	{
		consoleDebug(4, "Not improving on picked row {}: deltaAct={} oldViol = {} newViol = {}",
						pickedRow, deltaAct, oldViol, newViol);
		isCand = false;
	}
}

void WalkMIP::evalFlip(int var, int pickedRow, bool &isCand, double &damage)
{
	const Domain &domain = engine.getDomain();

	/* loop over affected rows and evaluate damage */
	isCand = true;
	damage = 0.0;
	double deltaX = (domain.lb(var) > 0.5) ? -1.0 : 1.0;
	for (const auto [row, coef] : mip.cols[var]) {
		evalFlipRow(row, coef, deltaX, pickedRow, isCand, damage);

		if (isCand == false);
			break;
	}

	const double obj_coef = mip.obj[var];
	if (isCand && !iszero(obj_coef))
		evalFlipRow(mip.nrows, obj_coef, deltaX, pickedRow, isCand, damage);


	consoleDebug(4, "Eval flip of {} for {} [cnt={}]: iscand = {} damage = {}",
				 deltaX, mip.cNames[var], mip.cols[var].size(), isCand, damage);
}

void WalkMIP::evalShiftRow(int row, double coef, double delta, int pickedRow, bool &isCand, double &damage) {
	const bool is_objective = (row == mip.nrows);
	const double rhs = is_objective ? engine.get_obj_rhs() : mip.rhs[row];
	const char sense = is_objective ? engine.get_obj_sense() : mip.sense[row];

	double minAct = engine.getMinAct(row);
	double maxAct = engine.getMaxAct(row);
	double deltaAct = delta * coef;
	double oldViol = rowViol(minAct, maxAct, sense, rhs);
	double newViol = rowViol(minAct + deltaAct, maxAct + deltaAct, sense, rhs);
	if (newViol > oldViol)
		damage += (newViol - oldViol);

	if ((row == pickedRow) && (newViol > oldViol))
	{
		consoleDebug(4, "Not improving on picked {}: deltaAct = {} oldViol = {} newViol = {}",
						pickedRow, deltaAct, oldViol, newViol);
		isCand = false;
	}
}

void WalkMIP::evalShift(int var, double delta, int pickedRow, bool &isCand, double &damage)
{
	const Domain &domain = engine.getDomain();

	/* loop over affected rows and evaluate damage */
	isCand = true;
	damage = 0.0;
	for (const auto [row, coef] : mip.cols[var]) {
		evalShiftRow(row, coef, delta, pickedRow, isCand, damage);

		if (isCand == false)
			break;
	}

	const double obj_coef = mip.obj[var];
	if (isCand && !iszero(obj_coef))
		evalShiftRow(mip.nrows, obj_coef, delta, pickedRow, isCand, damage);

	consoleDebug(4, "Eval shift of {} for {} [cnt={}]: iscand = {} damage = {}",
				 delta, mip.cNames[var], mip.cols[var].size(), isCand, damage);
}

void WalkMIP::walk()
{
	++n_walk;
	int n = mip.ncols;
	int m = mip.nrows;
	const auto &sense = mip.sense;
	const Domain &domain = engine.getDomain();

	// init best
	const IndexSet<int> &violated = engine.violatedRows();
	double bestViol = engine.violation();
	int bestNviol = violated.size();
	Domain::iterator bestMark = domain.mark();

	// main walk loop
	std::uniform_real_distribution<double> randomWalkDist(0.0, 1.0);
	int step = 0;
	score.resize(n);
	shifts.resize(n);
	int consecutiveNonBest = 0;
	int nSoftRestarts = 0;

	// tabu list to avoid short cycles
	IndexQueue<int> tabu(3, n);

	for (; step < maxRepairSteps; step++)
	{
		// Nothing to do if we got the violation to zero
		if (lessEqualThan(engine.violation(), ABS_FEASTOL))
			break;

		// Pick a violated constraint at random
		FP_ASSERT(!violated.empty());

		consoleDebug(3, "Repair step = {} #violated = {}", step, violated.size());

		std::uniform_int_distribution<int> uniformRow(0, violated.size() - 1);
		int violrow = violated[uniformRow(rndgen)];

		if (violrow == mip.nrows && !repair_objective)
			return;
			// consoleLog("Walking objective");

		const bool is_objective = (violrow == mip.nrows);

		const int *indices = is_objective ? mip.obj_cols.data() : mip.rows[violrow].idx();
		const double *coefs = is_objective ? mip.obj_coefs.data() : mip.rows[violrow].coef();
		const size_t cnt = is_objective ? mip.obj_cols.size() : mip.rows[violrow].size();

		const double rhs = is_objective ? engine.get_obj_rhs() : mip.rhs[violrow];
		const char sense = is_objective ? engine.get_obj_sense() : mip.sense[violrow];

		bool violatedLessThanSense = (sense == 'L') ||
									 ((sense == 'E') && (greaterThan(engine.getMinAct(violrow), rhs, ABS_FEASTOL)));

		consoleDebug(3, "Picked row {} violated {} act=[{},{}] rhs={}, violation={}", is_objective ? "objective" : mip.rNames[violrow],
					 violatedLessThanSense ? "<=" : ">=",
					 engine.getMinAct(violrow), engine.getMaxAct(violrow), rhs,
					 rowViol(engine.getMinAct(violrow), engine.getMaxAct(violrow), sense, rhs));

		// Retrieve list of candidates with corresponding score
		candidates.clear();
		double bestDamage = std::numeric_limits<double>::max();
		for (int pos = 0; pos < cnt; pos++)
		{
			int j = indices[pos];
			double coef = coefs[pos];
			if (domain.type(j) == 'C')
				break; //< vars are sorted by type

			/* Ignore non-fixed variables */
			if (domain.lb(j) < domain.ub(j))
				continue;

			/* Ignore tabu variables */
			if (tabu.has(j))
				continue;

			if (domain.type(j) == 'B')
			{
				FP_ASSERT((domain.ub(j) == 0.0) || (domain.lb(j) == 1.0));
				/* Only consider variables that can reduce violation of current constraint */
				bool posDeltaX = (domain.ub(j) < 0.5);
				double mult = (posDeltaX != violatedLessThanSense) ? +1.0 : -1.0;
				if (mult * coef < 0.0)
					continue;

				bool isCand;
				double damage;
				evalFlip(j, violrow, isCand, damage);

				if (isCand)
				{
					candidates.push_back(j);
					score[j] = damage;
					if (damage < bestDamage)
						bestDamage = damage;
				}
			}
			else
			{
				FP_ASSERT(domain.type(j) == 'I');
				/* Compute the (non necessarily integer) shift that would fix this constraint */
				double shift = 0.0;
				if (violatedLessThanSense)
					shift = (rhs - engine.getMinAct(violrow)) / coef;
				else
					shift = (rhs - engine.getMaxAct(violrow)) / coef;

				/* round it */
				if (shift > 0.0)
					shift = ceilEps(shift, ZEROTOL);
				else
					shift = floorEps(shift, ZEROTOL);

				if (isNull(shift, ZEROTOL))
					continue;

				/* Make sure we do not overshoot our own bounds */
				double newValue = domain.lb(j) + shift;
				newValue = std::min(newValue, std::min(mip.ub[j], domain.infinity / 2.0));
				newValue = std::max(newValue, std::max(mip.lb[j], -domain.infinity / 2.0));

				if (equal(domain.lb(j), newValue, ZEROTOL))
					continue;

				/* Now we can evaluate this shift */
				bool isCand;
				double damage;
				evalShift(j, newValue - domain.lb(j), violrow, isCand, damage);

				if (isCand)
				{
					candidates.push_back(j);
					score[j] = damage;
					shifts[j] = newValue - domain.lb(j);
					if (damage < bestDamage)
						bestDamage = damage;
				}
			}
		}

		consoleDebug(3, "candidates = {} / {}", candidates.size(), cnt);

		// It can happen that no flip can improve this constraint...skip it
		if (candidates.empty())
			continue;
		FP_ASSERT(!candidates.empty());

		int toFlip = mip.ncols;
		bool randomFlip = false;

		// If there is some damage, do a pure random walk flip with probability p
		if (isNotNull(bestDamage, ABS_FEASTOL) && randomWalkDist(rndgen) < randomWalkProbability)
		{
			std::uniform_int_distribution<int> uniformBin(0, candidates.size() - 1);
			toFlip = candidates[uniformBin(rndgen)];
			randomFlip = true;
		}
		else
		{
			// Otherwise, pick randomly from the candidates with the best damage
			// remove candidates with non lowest damage
			size_t first = 0;
			for (size_t k = 0; k < candidates.size(); k++)
			{
				if (lessEqualThan(score[candidates[k]], bestDamage, ABS_FEASTOL))
				{
					candidates[first] = candidates[k];
					first++;
				}
			}
			FP_ASSERT(first > 0);
			candidates.resize(first);
			std::uniform_int_distribution<int> uniformBin(0, candidates.size() - 1);
			toFlip = candidates[uniformBin(rndgen)];
		}
		FP_ASSERT(toFlip != mip.ncols);

		consoleDebug(3, "Candidates after filter = {} [bestDamage = {}, randomflip = {}]", candidates.size(), bestDamage, randomFlip);

		// Apply flip/shift
		if (domain.type(toFlip) == 'B')
			applyFlip(toFlip);
		else
			applyShift(toFlip, shifts[toFlip]);
		tabu.push(toFlip);

		// update best if possible
		if (lessEqualThan(engine.violation(), bestViol, ABS_FEASTOL))
		{
			consoleDebug(2, "Updated best: viol {} -> {}", bestViol, engine.violation());
			bestViol = engine.violation();
			bestNviol = violated.size();
			bestMark = domain.mark();
			consecutiveNonBest = 0;
			nSoftRestarts = 0;
		}
		else
			consecutiveNonBest++;

		// soft restart
		if (consecutiveNonBest >= maxRepairNonImprove)
		{
			consoleDebug(2, "Soft restart");
			engine.undo(bestMark);
			consecutiveNonBest = 0;
			nSoftRestarts++;
		}

		if (UserBreak)
			break;
		if (nSoftRestarts >= 500)
			break;
		if (gStopWatch().elapsed() >= timeLimit)
			break;
	}

	// Reset to best
	engine.undo(bestMark);
}

void WalkMIP::oneOpt()
{
	int n = mip.ncols;
	int m = mip.nrows;
	const auto &sense = mip.sense;

	// nothing to do if there is some violation
	if (!engine.violatedRows().empty())
		return;

	double objImpr = 0.0;
	int nShifted = 0;

	const Domain &domain = engine.getDomain();

	for (int j = 0; j < n; j++)
	{
		/* Cannot have unfixed variables */
		FP_ASSERT(equal(domain.lb(j), domain.ub(j), ABS_FEASTOL));

		/* Cannot improve objective is variable does not appear in there */
		if (isNull(mip.obj[j]))
			continue;

		double xj = domain.lb(j);
		bool posObj = (mip.objSense * mip.obj[j] > 0);
		bool canShift = true;
		double deltaUp = mip.ub[j] - xj;
		double deltaDown = xj - mip.lb[j];
		for (const auto [i, a] : mip.cols[j])
		{
			/* Cannot shift variables that appear in equality constraints */
			if (mip.sense[i] == 'E')
			{
				canShift = false;
				break;
			}

			/* All vars are fixed, so activities should match.
			 * However, because of incremental updates, they might be a bit off.
			 */
			FP_ASSERT(relEqual(engine.getMinAct(i), engine.getMaxAct(i), domain.zeroTol));

			/* Normalize to <= */
			double mult;
			double slack;
			if (mip.sense[i] == 'L')
			{
				mult = +1.0;
				slack = std::max(mip.rhs[i] - engine.getMaxAct(i), 0.0);
			}
			else
			{
				mult = -1.0;
				slack = std::max(engine.getMinAct(i) - mip.rhs[i], 0.0);
			}

			double coef = mult * a;
			if (coef > 0)
			{
				deltaUp = std::min(deltaUp, slack / coef);
			}
			else
			{
				deltaDown = std::min(deltaDown, slack / -coef);
			}
		}

		if (canShift)
		{
			deltaUp = std::max(deltaUp, 0.0);
			deltaDown = std::max(deltaDown, 0.0);
			if (domain.type(j) != 'C')
			{
				deltaUp = floorEps(deltaUp);
				deltaDown = floorEps(deltaDown);
			}

			double deltaX = posObj ? -deltaDown : deltaUp;
			if (isNotNull(deltaX, ZEROTOL))
			{
				engine.shift(j, xj + deltaX);
				objImpr += deltaX * mip.obj[j];
				nShifted++;
				FP_ASSERT(engine.violatedRows().empty());
			}
		}
	}

	// consoleLog("1-opt: {} flips improved objective by {}", nShifted, -objImpr);
}
