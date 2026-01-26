/**
 * @file mip.h
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

#pragma once

#include "cliquecover.h"
#include "cliquetable.h"
#include "impltable.h"
#include "lp_algorithm_type.h"
#include "solver_type.h"
#include "ranker_type.h"
#include "value_chooser_type.h"
#include "preset_type.h"

#include <maths.h>
#include <mipmodel.h>
#include <vector>

constexpr double INFTY = 1e20;
constexpr double REL_FEASTOL = 1e-6;

/* The two relevant tolerances for the MIP competition. */
constexpr double ABS_FEASTOL = 1e-6;
constexpr double ABS_INT_TOL = 1e-5;

constexpr double ZEROTOL = 1e-9;

/* MIP instance data */
struct MIPInstance
{
	int ncols = 0;
	int nrows = 0;
	double maxRhs = 0.0;
	// obj
	double objSense = 1.0;
	double objOffset = 0.0;
	std::vector<double> obj;
	// col data
	std::vector<char> xtype;
	std::vector<bool> is_integer;
	std::vector<double> lb;
	std::vector<double> ub;
	SparseMatrix cols;
	// row data
	std::vector<bool> is_equality;
	std::vector<char> sense;
	std::vector<double> rhs;
	SparseMatrix rows;
	// names (for debugging and output)
	std::vector<std::string> rNames;
	std::vector<std::string> cNames;
};

/* Extract instance data from a MIP model */
MIPInstance extract(MIPModelPtr model);

/* Input the LP relaxation of a given MIPInstance to the solver stored in MIPModelPtr. */
void pass_lp_to_solver(const MIPInstance& mip, MIPModelPtr model);

/* Checks whether a given solution vector x is feasible */
bool isSolFeasible(const MIPInstance &mip, std::span<const double> x);

/* Evaluate the objective value of a given solution vector */
double evalObj(const MIPInstance &mip, std::span<const double> x);

/* Evaluate gap between two bounds (normalized to [0,1], see primal integral paper */
inline double evalGap(double primalBound, double dualBound)
{
	const double epsZero = 1e-9;
	if (isNull(dualBound, epsZero) && isNull(primalBound, epsZero))
		return 0.0;
	if ((primalBound * dualBound) < 0.0)
		return 1.0;
	return fabs(primalBound - dualBound) / std::max(fabs(primalBound), fabs(dualBound));
}

/* Evaluate violation of constraint from activity bounds */
inline double rowViol(double minAct, double maxAct, char sense, double rhs)
{
	double viol = 0.0;
	if (sense == 'L')
		viol = std::max(minAct - rhs, 0.0);
	else if (sense == 'G')
		viol = std::max(rhs - maxAct, 0.0);
	else
		viol = std::max(minAct - rhs, rhs - maxAct);

	if (lessEqualThan(viol, ABS_FEASTOL))
		viol = 0.0;

	return viol;
}

/* Stores a complete MIP solution (feasible or not) */
struct Solution
{
public:
	std::vector<double> x;
	double objval;
	double absViolation;
	double relViolation;
	double timeFound;
	std::string foundBy; //< which algorithm found this solution

	/* Whether this is a partial solution. For partial solutions remaining x values are set to inf; objval, violation, and iFeas have no meaning when isPartial == true. */
	bool isPartial;
	bool isFeas;

	// equality operator
	bool operator==(const Solution &other) const
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
};
using SolutionPtr = std::shared_ptr<Solution>;

/* Construction a solution from a span range */
SolutionPtr makeFromSpan(const MIPInstance &mip, std::span<const double> x, double objval, bool isFeas = true, double violation = 0.0);

/* Stores a pool of MIP solutions (feasible or not).
 *
 * If not empty, the first solution is always the best one
 */
class SolutionPool
{
public:
	void setObjSense(double sense)
	{
		FP_ASSERT((sense == 1.0) || (sense == -1.0));
		objSense = sense;
	}
	void add(SolutionPtr sol);
	const std::vector<SolutionPtr> &getSols() const { return pool; }
	bool hasFeas() const
	{
		return ((!pool.empty()) && pool[0]->isFeas);
	}
	bool hasSols() const
	{
		return (!pool.empty());
	}
	SolutionPtr getIncumbent() const
	{
		return hasFeas() ? pool[0] : SolutionPtr();
	}
	double primalBound() const
	{
		return hasFeas() ? pool[0]->objval : objSense * INFTY;
	}
	double minViolation() const
	{
		FP_ASSERT(!pool.empty());
		return pool[0]->absViolation;
	}
	void merge(SolutionPool &other)
	{
		for (SolutionPtr sol : other.pool)
			add(sol);
		other.pool.clear();
		FP_ASSERT(!other.hasSols());
	}
	void print() const;

protected:
	double objSense = 1.0;
	std::vector<SolutionPtr> pool;
};

/* sort variables within each row by type ('B','I','C') and within each
 * subset by non-increasing absolute value of coefficients.
 */
void normalizeRows(MIPInstance &mip);

/* Row classification */
enum RowClass
{
	CLIQUE = 0,
	CLIQUE_EQ = 1,
	CLIQUE_EQ_N = 2,
	SETCOVER = 3,
	CARD = 4,
	CARD_EQ = 5,
	DOUBLE_AGGR = 6,
	VBOUND = 7,
	KNAPSACK = 8,
	KNAPSACK_EQ = 9,
	GENERIC = 10,
	NCLASSES
};

/* Get name of a row type */
inline std::string rClassName(RowClass rc)
{
	switch (rc)
	{
	case CLIQUE:
		return "CLIQUE";
	case CLIQUE_EQ:
		return "CLIQUE_EQ";
	case CLIQUE_EQ_N:
		return "CLIQUE_EQ_NEGATED";
	case SETCOVER:
		return "SETCOVER";
	case CARD:
		return "CARD";
	case CARD_EQ:
		return "CARD_EQ";
	case VBOUND:
		return "VBOUND";
	case DOUBLE_AGGR:
		return "DOUBLE_AGGR";
	case KNAPSACK:
		return "KNAPSACK";
	case KNAPSACK_EQ:
		return "KNAPSACK_EQ";
	case GENERIC:
		return "GENERIC";
	default:
		return "(unknown)";
	}
	return "(unknown)";
}

/* Parameters */
struct Params
{
public:
	/* Global parameters */
	uint64_t seed = 20250101;
	double timeLimit = 1200;
	int threads = 32;
	int maxTries = 1;

	/* Output. */
	bool enableOutput = true;
	int displayInterval = 500;

	/* Depth-first-search parameters. */
	bool propagate = true;
	bool repair = false;
	bool backtrackOnInfeas = true;
	double maxConsecutiveInfeas = 0.2; /** node limit as fraction of variables */

	int minNodes = 100000;
	int maxNodes = -1;
	int maxLpSolved = 1;
	int maxSolutions = 1;

	/* Strategies. */
	PresetType preset = PresetType::UNKNOWN;
	bool useOldBranching = false; /** Whether to use the old or the new branching strategy. */
	RankerType ranker = RankerType::TYPE;
	ValueChooserType valueChooser = ValueChooserType::RANDOM_LP;
	LpAlgorithmType lpMethod = LpAlgorithmType::BARRIER;
	LpAlgorithmType lpMethodFinal = LpAlgorithmType::BARRIER;
	bool mipPresolve = true; /** Whether to presolver the problem. */
	bool postsolve = false;	 /** Whether to postsolve found solutions. */
	bool writeSol = false;  /** Whether to print the final solution. */
	bool zeroObj = false;	 /** Whether to zero out the objective of the LP relaxation. */

	/* repair limits */
	double randomWalkProbability = 0.75;
	int maxRepairNonImprove = 10;
	int maxRepairSteps = 100;

	/* Solver settings. */
	SolverType solver = SolverType::COPT;
	SolverType presolver = SolverType::GUROBI;
	double lpTol = 1e-6;

	/* Whether to run all heuristics in parallel (after the LP solve). */
	bool runPortfolio = false;

	/* Private parameters (derived from input). */
	bool solveLp = false; /** Whether to solve the LP relaxation. */

	void readConfig();
	void logToConsole();
	void printUsage();
};

/* Global data structures on a MIP instance */
struct MIPData
{
	MIPData(MIPModelPtr model, bool build_clique_cover);
	MIPData(MIPInstance&& mip, MIPModelPtr lp_solver, bool build_clique_cover);

	// MIP instance
	MIPInstance mip;
	// global structures
	std::vector<RowClass> rclass; //< row classification
	std::vector<int> uplocks;	  //< up locks
	std::vector<int> dnlocks;	  //< down locks
	CliqueTable cliquetable;
	ImplTable impltable;
	CliqueCover cliquecover;
	// stats
	int nBinaries;
	int nIntegers;
	int nContinuous;
	int numSingletons;
	int objSupport;
	// solution and bounds
	double dualBound;
	SolutionPool solpool;
	// relaxations
	MIPModelPtr lp; //< LP relaxation model

	/* Solution vectors obtained by either Simplex/Bar/FOM.*/
	std::vector<double> primals;
	std::vector<double> duals;
	std::vector<double> reduced_costs;
};

/* classify rows depending on structure
 *
 * @note: In general, there can ambiguity for uniform binary rows:
 *        For example, x1 + x2 <= 1 can be either classified as a clique constraint
 *        or as a set covering constraint on the negated literals
 *        (1-x1) + (1-x2) >= 1.
 *        Similarly, x1 + x2 + x3 <= 1 is a cardinality constraint, but also
 *        a set covering on negated literals (1-x1) + (1-x2) + (1-x3) >= 1.
 */
RowClass classifyRow(SparseVector::view_type row, char sense, double rhs, std::span<const char> xtype);

void rowClassification(MIPData &data);

// compute variable locks
void computeColLocks(MIPData &data);

// column statistics
void colStats(MIPData &data);

// Extract cliques from instance
void constructCliquetable(MIPData &data);

// Extract implications from instance
void constructImpltable(MIPData &data);

// Construct a cliquecover over the binary variables
void constructCliqueCover(MIPData &data);

// Solve LP relaxation
std::vector<double> solveLP(MIPModelPtr model, const Params &params, bool enableOutput);
