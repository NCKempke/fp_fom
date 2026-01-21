/**
 * @file mip.cpp
 * @brief MIP related data structures and methods
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 * @contributor Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2021-2025
 *
 * Copyright 2021 Domenico Salvagnin
 * Copyright 2025 Nils-Christian Kempke
 */

#include "mip.h"
#include <consolelog.h>
#include <fileconfig.h>
#include <timer.h>
#include <maths.h>

MIPInstance extract(MIPModelPtr model)
{
	MIPInstance mip;
	mip.ncols = model->ncols();
	mip.nRows = model->nrows();
	// get obj
	mip.objSense = (model->objSense() == ObjSense::MIN) ? 1.0 : -1.0;
	mip.objOffset = model->objOffset();
	mip.obj.resize(mip.ncols);
	model->objcoefs(&(mip.obj[0]));
	// get col data
	mip.xtype.resize(mip.ncols);
	model->ctypes(&(mip.xtype[0]));
	mip.lb.resize(mip.ncols);
	model->lbs(&(mip.lb[0]));
	mip.ub.resize(mip.ncols);
	model->ubs(&(mip.ub[0]));
	model->cols(mip.cols);
	// get row data
	mip.sense.resize(mip.nRows);
	model->sense(&(mip.sense[0]));
	mip.rhs.resize(mip.nRows);
	model->rhs(&(mip.rhs[0]));
	model->rows(mip.rows);
	// names
	model->rowNames(mip.rNames);
	model->colNames(mip.cNames);

	for (const auto &rhs : mip.rhs)
		mip.maxRhs = std::max(std::abs(rhs), mip.maxRhs);
	return mip;
}

/* Checks whether a given solution vector x is feasible */
bool isSolFeasible(const MIPInstance &mip, std::span<const double> x)
{
	FP_ASSERT(x.size() >= mip.ncols);

	/* Check bounds and integrality with ABS_FEASTOL. */
	for (int j = 0; j < mip.ncols; j++)
	{
		if (lessThan(x[j], mip.lb[j], ABS_FEASTOL))
			return false;
		if (greaterThan(x[j], mip.ub[j], ABS_FEASTOL))
			return false;
		if ((mip.xtype[j] != 'C') && !isInteger(x[j]))
			return false;
	}

	/* Check constraints with ABS and REL feas tol (since we are usually not running Simplex and cannot hope that the barrier solution will be feasible in absolute tolerances). */
	for (int i = 0; i < mip.nRows; i++)
	{
		// compute violation
		double viol = dotProduct(mip.rows[i], x.data()) - mip.rhs[i];
		if (mip.sense[i] == 'G')
		{
			viol = -viol;
		}
		else if (mip.sense[i] == 'L')
		{ /* nothing to do */
		}
		else if (mip.sense[i] == 'E')
		{
			viol = fabs(viol);
		}
		else
		{
			FP_ASSERT(false);
		}

		if (isPositive(viol, ABS_FEASTOL) && isPositive(viol, mip.maxRhs * REL_FEASTOL))
		{
			return false;
		}
	}

	return true;
}

/* Evaluate the objective value of a given solution vector */
double evalObj(const MIPInstance &mip, std::span<const double> x)
{
	FP_ASSERT((int)x.size() >= mip.ncols);
	return dotProduct(mip.obj.data(), x.data(), mip.ncols) + mip.objOffset;
}

SolutionPtr makeFromSpan(const MIPInstance &mip, std::span<const double> x, double objval, bool isFeas, double violation)
{
	SolutionPtr sol = std::make_shared<Solution>();
	sol->x.insert(sol->x.end(), x.begin(), x.end());
	sol->objval = objval;
	FP_ASSERT(equal(sol->objval, evalObj(mip, sol->x)));
	sol->isFeas = isFeas;
	FP_ASSERT(sol->isFeas == isSolFeasible(mip, sol->x));
	sol->absViolation = violation;
	sol->relViolation = violation / mip.maxRhs;

	return sol;
}

void SolutionPool::add(SolutionPtr sol)
{
	if (!sol)
		return;

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
				return (objSense * sol1->objval < objSense * sol2->objval);
			else
				return (sol1->absViolation < sol2->absViolation);
		}
		return (sol1->isFeas);
	};
	pool.insert(
		std::upper_bound(pool.begin(), pool.end(), sol, comp),
		sol);

	// 20 solutions should be enough for our purposes
	if (pool.size() > 20)
		pool.resize(20);
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

/* sort variables in a row by type ('B','I','C') and within each
 * subset by non-increasing absolute value of coefficients.
 */
static void normalizeRow(MIPInstance &mip, int *vars, double *coefs, int count)
{
	/* First bucket sort by type */
	int numBin = 0;
	int numInt = 0;
	for (int k = 0; k < count; k++)
	{
		if (mip.xtype[vars[k]] == 'B')
			numBin++;
		else if (mip.xtype[vars[k]] == 'I')
			numInt++;
	}

	int startBin = 0;
	int startInt = numBin;
	int startCont = numBin + numInt;
	std::vector<std::pair<int, double>> row(count);
	for (int k = 0; k < count; k++)
	{
		if (mip.xtype[vars[k]] == 'B')
			row[startBin++] = {vars[k], coefs[k]};
		else if (mip.xtype[vars[k]] == 'I')
			row[startInt++] = {vars[k], coefs[k]};
		else
			row[startCont++] = {vars[k], coefs[k]};
	}
	FP_ASSERT(startBin == numBin);
	FP_ASSERT(startInt == (numBin + numInt));

	/* Now sort by abs(coefficient) within each type */
	auto cmp = [](const auto &e1, const auto &e2)
	{
		return (fabs(e1.second) > fabs(e2.second));
	};
	std::sort(row.begin(), row.begin() + numBin, cmp);
	std::sort(row.begin() + numBin, row.begin() + numBin + numInt, cmp);
	std::sort(row.begin() + numBin + numInt, row.end(), cmp);

	/* Copy over original row */
	for (int k = 0; k < count; k++)
	{
		vars[k] = row[k].first;
		coefs[k] = row[k].second;
	}

	/* Make sure everything is as expected */
	for (int k = 0; k < (numBin - 1); k++)
	{
		FP_ASSERT(mip.xtype[vars[k]] == 'B');
		FP_ASSERT(fabs(coefs[k]) >= fabs(coefs[k + 1]));
	}
	for (int k = numBin; k < (numBin + numInt - 1); k++)
	{
		FP_ASSERT(mip.xtype[vars[k]] == 'I');
		FP_ASSERT(fabs(coefs[k]) >= fabs(coefs[k + 1]));
	}
	for (int k = numBin + numInt; k < count - 1; k++)
	{
		FP_ASSERT(mip.xtype[vars[k]] == 'C');
		FP_ASSERT(fabs(coefs[k]) >= fabs(coefs[k + 1]));
	}
}

void normalizeRows(MIPInstance &mip)
{
	int m = mip.nRows;
	for (int i = 0; i < m; i++)
	{
		int *vars = &(mip.rows.ind[mip.rows.beg[i]]);
		double *coefs = &(mip.rows.val[mip.rows.beg[i]]);
		normalizeRow(mip, vars, coefs, mip.rows.cnt[i]);
	}
}

#define READ_ASSIGN_PARAM(name) name = gConfig().get("" #name, defaults.name)
#define READ_PARAM(name) gConfig().getValueString("" #name)

void Params::readConfig()
{
	bool error = false;

	Params defaults;
	READ_ASSIGN_PARAM(seed);
	READ_ASSIGN_PARAM(timeLimit);
	READ_ASSIGN_PARAM(threads);
	READ_ASSIGN_PARAM(maxTries);

	READ_ASSIGN_PARAM(enableOutput);
	READ_ASSIGN_PARAM(displayInterval);

	READ_ASSIGN_PARAM(propagate);
	READ_ASSIGN_PARAM(repair);
	READ_ASSIGN_PARAM(backtrackOnInfeas);
	READ_ASSIGN_PARAM(maxConsecutiveInfeas);

	READ_ASSIGN_PARAM(minNodes);
	READ_ASSIGN_PARAM(maxNodes);
	READ_ASSIGN_PARAM(maxLpSolved);
	READ_ASSIGN_PARAM(maxSolutions);

	READ_ASSIGN_PARAM(mipPresolve);
	READ_ASSIGN_PARAM(postsolve);
	READ_ASSIGN_PARAM(writeSol);
	READ_ASSIGN_PARAM(zeroObj);
	READ_ASSIGN_PARAM(randomWalkProbability);
	READ_ASSIGN_PARAM(maxRepairNonImprove);
	READ_ASSIGN_PARAM(maxRepairSteps);
	READ_ASSIGN_PARAM(lpTol);

	READ_ASSIGN_PARAM(runPortfolio);

	/* Read solver and strategy info. */
	READ_ASSIGN_PARAM(useOldBranching);
	std::string presetConfig = READ_PARAM(preset);
	std::string solverConfig = READ_PARAM(solver);
	std::string presolverConfig = READ_PARAM(presolver);
	std::string lpMethodConfig = READ_PARAM(lpMethod);
	std::string lpMethodFinalConfig = READ_PARAM(lpMethodFinal);
	std::string rankerConfig = READ_PARAM(ranker);
	std::string valueChooserConfig = READ_PARAM(valueChooser);

	preset = presetConfig.empty() ? defaults.preset : PresetTypeFromString(presetConfig);
	lpMethod = lpMethodConfig.empty() ? defaults.lpMethod : LpAlgorithmTypeFromString(lpMethodConfig);

	if (preset != PresetType::UNKNOWN)
	{
		const auto [rankerPreset, chooserPreset] = getRankerAndValueChooserFromPreset(preset);
		consoleLog("Preset={}; overwriting potentially specified ranker and valueChooser:", toString(preset));
		consoleLog("\tranker={} valueChooser={}", toString(rankerPreset), toString(chooserPreset));

		ranker = rankerPreset;
		valueChooser = chooserPreset;

		if (preset == PresetType::ZEROCORE)
		{
			/* Barrier zero obj */
			consoleLog("\tlpSolver={}, zeroObj=true\n", toString(LpAlgorithmType::BARRIER));
			lpMethod = LpAlgorithmType::BARRIER;
			zeroObj = true;
		}
		else if (preset == PresetType::ZEROLP)
		{
			/* Simplex zero obj */
			consoleLog("\tlpSolver={}, zeroObj=true\n", toString(LpAlgorithmType::DUAL_SIMPLEX));
			lpMethod = LpAlgorithmType::DUAL_SIMPLEX;
			zeroObj = true;
		}
		else if (preset == PresetType::CORE)
		{
			/* Barrier */
			consoleLog("\tlpSolver={}\n", toString(LpAlgorithmType::BARRIER));
			lpMethod = LpAlgorithmType::BARRIER;
		}
		else if (preset == PresetType::LP)
		{
			/* Simplex */
			consoleLog("\tlpSolver={}\n", toString(LpAlgorithmType::DUAL_SIMPLEX));
			lpMethod = LpAlgorithmType::DUAL_SIMPLEX;
		}
	}
	else
	{
		valueChooser = valueChooserConfig.empty() ? defaults.valueChooser : ValueChooserTypeFromString(valueChooserConfig);
		ranker = rankerConfig.empty() ? defaults.ranker : RankerTypeFromString(rankerConfig);
	}

	lpMethodFinal = lpMethodFinalConfig.empty() ? defaults.lpMethodFinal : LpAlgorithmTypeFromString(lpMethodFinalConfig);

	solver = solverConfig.empty() ? defaults.solver : SolverTypeFromString(solverConfig);
	presolver = presolverConfig.empty() ? defaults.presolver : SolverTypeFromString(presolverConfig);

	solveLp = rankerNeedsLpSolve(ranker) || valueChooserNeedsLpSolve(valueChooser) || runPortfolio;

	if (!presetConfig.empty() && preset == PresetType::UNKNOWN)
	{
		consoleError("Unknown preset {}; aborting", presetConfig);
		printPresets();
		error = true;
	}

	if (ranker == RankerType::UNKNOWN)
	{
		consoleError("Unknown variable select type {}; aborting", rankerConfig);
		printRankers();
		error = true;
	}

	if (valueChooser == ValueChooserType::UNKNOWN)
	{
		consoleError("Unknown value chooser type {}; aborting", rankerConfig);
		printValueChoosers();
		error = true;
	}

	if (lpMethod == LpAlgorithmType::UNKNOWN)
	{
		consoleError("Unknown lp algorithm {}; aborting", lpMethodConfig);
		printLpMethods();
		error = true;
	}

	if (lpMethodFinal == LpAlgorithmType::UNKNOWN)
	{
		consoleError("Unknown final lp algorithm {}; aborting", lpMethodFinalConfig);
		printLpMethods();
		error = true;
	}

	if (solver == SolverType::UNKNOWN)
	{
		consoleError("Unknown solver {}; aborting", solverConfig);
		printSolverTypes();
		error = true;
	}

	if (presolver == SolverType::UNKNOWN)
	{
		consoleError("Unknown presolver {}; aborting", presolverConfig);
		printSolverTypes();
		error = true;
	}

	if (presolver != SolverType::CPLEX && presolver != SolverType::GUROBI)
	{
		consoleError("Only CPLEX and GUROBI are available for presolve; aborting");
		error = true;
	}

	if (presolver != SolverType::CPLEX && postsolve)
	{
		consoleError("Only CPLEX offers mip postsolve! Deactivating postsolve.");
		postsolve = 0;
	}

	if (lpMethod == LpAlgorithmType::FIRST_ORDER_METHOD && solver != SolverType::COPT)
	{
		consoleError("Only solver={} is available for lpMethod={}", toString(SolverType::COPT), toString(LpAlgorithmType::FIRST_ORDER_METHOD));
		error = true;
	}

	if (error)
		exit(1);
}

#define LOG_PARAM_DEFAULT(name) consoleLog(#name "={}", name)

void Params::printUsage()
{
	consoleLog("Parameters and their defaults:");
	consoleLog("");

	LOG_PARAM_DEFAULT(seed);
	LOG_PARAM_DEFAULT(timeLimit);
	LOG_PARAM_DEFAULT(threads);
	LOG_PARAM_DEFAULT(maxTries);

	LOG_PARAM_DEFAULT(enableOutput);
	LOG_PARAM_DEFAULT(displayInterval);

	LOG_PARAM_DEFAULT(propagate);
	LOG_PARAM_DEFAULT(repair);
	LOG_PARAM_DEFAULT(backtrackOnInfeas);
	LOG_PARAM_DEFAULT(maxConsecutiveInfeas);

	LOG_PARAM_DEFAULT(minNodes);
	LOG_PARAM_DEFAULT(maxNodes);
	LOG_PARAM_DEFAULT(maxLpSolved);
	LOG_PARAM_DEFAULT(maxSolutions);

	LOG_PARAM_DEFAULT(useOldBranching);
	consoleLog("preset={}", toString(preset));
	consoleLog("ranker={}", toString(ranker));
	consoleLog("valueChooser={}", toString(valueChooser));
	consoleLog("lpMethod={}", toString(lpMethod));
	consoleLog("lpMethodFinal={}", toString(lpMethodFinal));
	LOG_PARAM_DEFAULT(mipPresolve);
	LOG_PARAM_DEFAULT(postsolve);
	LOG_PARAM_DEFAULT(writeSol);
	LOG_PARAM_DEFAULT(zeroObj);
	LOG_PARAM_DEFAULT(randomWalkProbability);
	LOG_PARAM_DEFAULT(maxRepairNonImprove);
	LOG_PARAM_DEFAULT(maxRepairSteps);

	consoleLog("solver={}", toString(solver));
	consoleLog("presolver={}", toString(presolver));
	LOG_PARAM_DEFAULT(lpTol);

	LOG_PARAM_DEFAULT(runPortfolio);

	consoleLog("");
	consoleLog("Available options:");
	consoleLog("");

	printRankers();
	consoleLog("");

	printValueChoosers();
	consoleLog("");

	printLpMethods();
	consoleLog("");

	printSolverTypes();
	consoleLog("");

	consoleLog("Presolve is available with {} and {}", toString(SolverType::GUROBI), toString(SolverType::CPLEX));
	consoleLog("Postsolve is available only with {}", toString(SolverType::CPLEX));
}

#define LOG_PARAM(name) consoleLog(#name " = {}", name)

void Params::logToConsole()
{
	LOG_PARAM(seed);
	LOG_PARAM(timeLimit);
	LOG_PARAM(threads);
	LOG_PARAM(maxTries);

	LOG_PARAM(enableOutput);
	LOG_PARAM(displayInterval);

	LOG_PARAM(propagate);
	LOG_PARAM(repair);
	LOG_PARAM(backtrackOnInfeas);
	LOG_PARAM(maxConsecutiveInfeas);

	LOG_PARAM(minNodes);
	LOG_PARAM(maxNodes);
	LOG_PARAM(maxLpSolved);
	LOG_PARAM(maxSolutions);

	LOG_PARAM(useOldBranching);
	consoleLog("preset={}", toString(preset));
	consoleLog("ranker = {}", toString(ranker));
	consoleLog("valueChooser = {}", toString(valueChooser));
	consoleLog("lpMethod = {}", toString(lpMethod));
	consoleLog("lpMethodFinal = {}", toString(lpMethodFinal));
	LOG_PARAM(mipPresolve);
	LOG_PARAM(postsolve);
	LOG_PARAM(writeSol);
	LOG_PARAM(randomWalkProbability);
	LOG_PARAM(maxRepairNonImprove);
	LOG_PARAM(maxRepairSteps);
	LOG_PARAM(zeroObj);

	consoleLog("solver = {}", toString(solver));
	consoleLog("presolver = {}", toString(presolver));
	LOG_PARAM(lpTol);

	LOG_PARAM(runPortfolio);
}

// classify a single row
RowClass classifyRow(SparseVector::view_type row, char sense, double rhs, std::span<const char> xtype)
{
	FP_ASSERT(row.size()); //< no empty rows please!
	// compute stats for this row
	int posbincnt = 0;
	int negbincnt = 0;
	int othercnt = 0;
	double minAbs = INFTY;
	double maxAbs = 0.0;
	for (const auto &[j, v] : row)
	{
		if (xtype[j] != 'B')
		{
			othercnt++;
		}
		else
		{
			double absV = fabs(v);
			minAbs = std::min(minAbs, absV);
			maxAbs = std::max(maxAbs, absV);
			if (v > 0.0)
				posbincnt++;
			else
				negbincnt++;
		}
	}

	// use them for classification
	RowClass ret = GENERIC;

	if (othercnt)
	{
		// non all-binary row
		if ((othercnt == 1) && ((posbincnt + negbincnt) == 1))
		{
			if (sense == 'E')
				ret = DOUBLE_AGGR;
			else
				ret = VBOUND;
		}
		else
		{
			ret = GENERIC;
		}
	}
	else
	{
		// all binary row
		if (greaterThan(maxAbs, minAbs, ZEROTOL))
		{
			// knapsack types
			if (sense == 'E')
				ret = KNAPSACK_EQ;
			else
				ret = KNAPSACK;
		}
		else
		{
			// uniform binary row: (generalized) setcover/setpartition/setpacking/cardinality constraints
			FP_ASSERT(equal(minAbs, maxAbs, ZEROTOL));
			double factor = minAbs;
			if (sense == 'L')
			{
				if (equal(factor * (1 - negbincnt), rhs, ZEROTOL))
					ret = CLIQUE;
				else if (equal(factor * (posbincnt - 1), rhs, ZEROTOL))
					ret = SETCOVER;
				else
					ret = CARD;
			}
			else if (sense == 'G')
			{
				if (equal(factor * (1 - negbincnt), rhs, ZEROTOL))
					ret = SETCOVER;
				else if (equal(factor * (posbincnt - 1), rhs, ZEROTOL))
					ret = CLIQUE;
				else
					ret = CARD;
			}
			else if (sense == 'E')
			{
				if (equal(factor * (1 - negbincnt), rhs, ZEROTOL))
					ret = CLIQUE_EQ;
				else if (equal(factor * (posbincnt - 1), rhs, ZEROTOL))
					ret = CLIQUE_EQ_N;
				else
					ret = CARD_EQ;
			}
			else
			{
				FP_ASSERT(false);
			}
		}
	}
	return ret;
}

// classify rows depending on structure
void rowClassification(MIPData &data)
{
	const MIPInstance &mip = data.mip;
	int m = mip.nRows;
	int n = mip.ncols;
	data.rclass.resize(m);

	// classify rows
	for (int i = 0; i < m; i++)
	{
		data.rclass[i] = classifyRow(mip.rows[i], mip.sense[i], mip.rhs[i], mip.xtype);
	}

	// print stats
	std::vector<int> counts(RowClass::NCLASSES, 0);
	for (int i = 0; i < m; i++)
		counts[data.rclass[i]]++;
	consoleLog("Row classification:");
	for (int rc = 0; rc < (int)RowClass::NCLASSES; rc++)
		consoleLog("{}: {}", rClassName((RowClass)rc), counts[rc]);
	consoleLog("");
}

// compute variable locks
void computeColLocks(MIPData &data)
{
	const MIPInstance &mip = data.mip;
	int n = mip.ncols;
	int m = mip.nRows;
	data.uplocks.resize(n);
	data.dnlocks.resize(n);
	std::fill(data.uplocks.begin(), data.uplocks.end(), 0);
	std::fill(data.dnlocks.begin(), data.dnlocks.end(), 0);

	for (int i = 0; i < m; i++)
	{
		const auto &row = mip.rows[i];
		for (const auto &[j, v] : row)
		{
			if (mip.sense[i] == 'E')
			{
				data.uplocks[j]++;
				data.dnlocks[j]++;
			}
			else
			{
				double mult = (mip.sense[i] == 'L') ? 1.0 : -1.0;
				if ((mult * v) > 0.0)
					data.uplocks[j]++;
				else
					data.dnlocks[j]++;
			}
		}
	}
}

void colStats(MIPData &data)
{
	const MIPInstance &mip = data.mip;
	int numBin = 0;
	int numInt = 0;
	int numCont = 0;
	int numSingletons = 0;
	double minLB = INFTY;	//< for integer variables only
	double maxUB = 0.0;		//< for integer variables only
	double maxDomain = 0.0; //< for integer variables only
	int dnRoundable = 0;
	int upRoundable = 0;
	int objSupport = 0;

	// get variable data
	int n = mip.ncols;

	// collects stats
	for (int j = 0; j < n; j++)
	{
		if (isNotNull(mip.obj[j]))
			objSupport++;

		const auto &col = mip.cols[j];
		if (col.size() <= 1)
			numSingletons++;

		if (!data.uplocks[j])
			upRoundable++;
		if (!data.dnlocks[j])
			dnRoundable++;

		if (mip.xtype[j] == 'B')
		{
			numBin++;
		}
		else if (mip.xtype[j] == 'I')
		{
			numInt++;
			minLB = std::min(minLB, mip.lb[j]);
			maxUB = std::max(maxUB, mip.ub[j]);
			maxDomain = std::max(maxDomain, mip.ub[j] - mip.lb[j]);
		}
		else if (mip.xtype[j] == 'C')
		{
			numCont++;
		}
		else
		{
			FP_ASSERT(false);
		}
	}

	if (!numInt)
		minLB = 0.0;

	data.nBinaries = numBin;
	data.nIntegers = numInt;
	data.nContinuous = numCont;
	data.numSingletons = numSingletons;
	data.objSupport = objSupport;

	// print stats
	consoleLog("Col classification:");
	consoleLog("BINARIES: {}", numBin);
	consoleLog("INTEGERS: {} [{},{}] |{}|", numInt, minLB, maxUB, maxDomain);
	consoleLog("CONTINUOUS: {}", numCont);
	consoleLog("SINGLETONS: {}", numSingletons);
	consoleLog("OBJSUPPORT: {}", objSupport);
	consoleLog("UPROUNDABLE: {}", upRoundable);
	consoleLog("DNROUNDABLE: {}", dnRoundable);
	consoleLog("");
}

/* Extract cliques from instance */
void constructCliquetable(MIPData &data)
{
	const MIPInstance &mip = data.mip;
	data.cliquetable.setNcols(mip.ncols);
	FP_ASSERT(data.cliquetable.nCliques() == 0);
	FP_ASSERT(data.cliquetable.nNonzeros() == 0);

	int m = mip.nRows;
	int n = mip.ncols;
	std::vector<int> clique;

	for (int i = 0; i < m; i++)
	{
		if ((data.rclass[i] == CLIQUE) ||
			(data.rclass[i] == CLIQUE_EQ) ||
			(data.rclass[i] == CLIQUE_EQ_N))
		{
			// This is a clique row straight from the matrix
			clique.clear();
			double mult = 1.0;
			// We need to figure out whether this is a negated clique or not
			if (data.rclass[i] == CLIQUE)
			{
				mult = (mip.sense[i] == 'G') ? -1.0 : +1.0;
			}
			else
			{
				mult = (data.rclass[i] == CLIQUE_EQ) ? 1.0 : -1.0;
			}
			const auto &row = mip.rows[i];
			for (const auto &[j, v] : row)
			{
				if (mult * v > 0.0)
					clique.push_back(posLit(j, n));
				else
					clique.push_back(negLit(j, n));
			}
			data.cliquetable.add(clique, mip.sense[i] == 'E');
		}
		/* TODO: we could extract cliques from knapsack rows, for example... */
	}

	data.cliquetable.constructLitWiseRepr();

	consoleLog("Cliquetable: {} cliques and {} nonzeros", data.cliquetable.nCliques(), data.cliquetable.nNonzeros());
}

/* Extract implications from instance */
void constructImpltable(MIPData &data)
{
	const MIPInstance &mip = data.mip;
	data.impltable.setNcols(mip.ncols);
	FP_ASSERT(data.impltable.nImpls() == 0);

	int m = mip.nRows;
	int n = mip.ncols;

	for (int i = 0; i < m; i++)
	{
		if (data.rclass[i] == VBOUND)
		{
			const auto &row = mip.rows[i];
			FP_ASSERT(mip.sense[i] != 'E');
			FP_ASSERT(row.size() == 2);
			const int *vars = row.idx();
			const double *coefs = row.coef();
			FP_ASSERT(mip.xtype[vars[0]] == 'B');
			FP_ASSERT(mip.xtype[vars[1]] != 'B');
			double bincoef = coefs[0];
			double othercoef = coefs[1];
			char sense = mip.sense[i];
			double rhs = mip.rhs[i];
			bincoef /= othercoef;
			rhs /= othercoef;
			/* flip sense if we divided by a negative number */
			if (othercoef < 0.0)
				sense = (sense == 'L') ? 'G' : 'L';
			if (sense == 'L')
			{
				// we get an implied upper bound
				double bound = mip.ub[vars[1]];
				// for x = 0 the new upper bound is rhs
				if (lessThan(rhs, bound, ZEROTOL))
					data.impltable.add(vars[0], false, vars[1], true, rhs);
				// for x = 1 the new upper bound is rhs-bincoef
				if (lessThan(rhs - bincoef, bound, ZEROTOL))
					data.impltable.add(vars[0], true, vars[1], true, rhs - bincoef);
			}
			else
			{
				// we get an implied lower bound
				double bound = mip.lb[vars[1]];
				// for x = 0 the new lower bound is rhs
				if (greaterThan(rhs, bound, ZEROTOL))
					data.impltable.add(vars[0], false, vars[1], false, rhs);
				// for x = 1 the new lower bound is rhs-bincoef
				if (greaterThan(rhs - bincoef, bound, ZEROTOL))
					data.impltable.add(vars[0], true, vars[1], false, rhs - bincoef);
			}
		}
		/* TODO: what about double_aggr and other constraints? */
	}

	data.impltable.sort();

	consoleLog("Impltable: {} implications", data.impltable.nImpls());
}

// Construct a cliquecover over the binary variables
void constructCliqueCover(MIPData &data)
{
	// clique covers
	std::vector<int> binaries;
	for (int j = 0; j < data.mip.ncols; j++)
	{
		if (data.mip.xtype[j] == 'B')
			binaries.push_back(j);
	}
	data.cliquecover = greedyCliqueCover(data.cliquetable, binaries, false);

	consoleLog("Clique cover: {} cliques, {} / {}", data.cliquecover.nCliques(), data.cliquecover.nCovered(), binaries.size());
}

std::vector<double> solveLP(MIPModelPtr model, const Params &params, bool enableOutput)
{
	std::vector<double> x;
	model->logging(enableOutput);
	model->handleCtrlC(true);
	model->seed(params.seed);
	model->dblParam(DblParam::TimeLimit, std::max(params.timeLimit - gStopWatch().elapsed(), 0.0));
	model->intParam(IntParam::Threads, params.threads);

	model->lpopt(solverChar(params.lpMethod), params.lpTol, params.lpTol);

	model->handleCtrlC(false);
	model->logging(false);

	// TODO this is weird?
	if (model->isPrimalFeas())
	{
		x.resize(model->ncols());
		model->sol(x.data());
	}

	return x;
}

// Init MIP Data from a MIP model
MIPData::MIPData(MIPModelPtr model, bool build_clique_cover)
{
	mip = extract(model);
	solpool.setObjSense(mip.objSense);
	dualBound = -INFTY * mip.objSense;

	// normalize rows
	normalizeRows(mip);

	// row classification
	rowClassification(*this);

	// variable locks
	computeColLocks(*this);
	colStats(*this);

	// global tables
	constructCliquetable(*this);
	constructImpltable(*this);

	if (build_clique_cover)
		constructCliqueCover(*this);
}
