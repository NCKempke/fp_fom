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
#include "solpool.h"
#include "solver_type.h"
#include "tolerances.h"
#include "ranker_type.h"
#include "value_chooser_type.h"
#include "preset_type.h"

#include <maths.h>
#include <mipmodel.h>

#include <atomic>
#include <vector>

#include "consolelog.h"

/* MIP instance data */
struct MIPInstance
{
	/* Columns are stored [binaries, integers, continuous]. */
	int ncols = 0;
	int n_binaries = 0;
	int n_integers = 0;

	/* Rows are stored [equalities, inequalities]. */
	int nrows = 0;
	int n_equalities = 0;
	double maxRhs = 0.0;
	// obj
	double objSense = 1.0;
	double objOffset = 0.0;

	/* Column data. */
	std::vector<double> obj;
	std::vector<char> xtype;
	std::vector<double> lb;
	std::vector<double> ub;

	/* Sparse Objective info. */
	std::vector<double> obj_coefs;
	std::vector<int> obj_cols;

	SparseMatrix cols;
	// row data
	std::vector<char> sense;
	std::vector<double> rhs;
	SparseMatrix rows;

	// names (for debugging and output)
	std::vector<std::string> rNames;
	std::vector<std::string> cNames;
	/* Inverse column permutation for solution output; maps orig_col -> col. */
	std::vector<int> map_orig_to_new_col;
};

/* Extract instance data from a MIP model */
MIPInstance extract(MIPModelPtr model);

/* Construction a solution from a span range */
std::unique_ptr<Solution> makeFromSpan(const MIPInstance &mip, std::span<const double> x, double objval, bool isFeas = true, double violation = 0.0);

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

inline void printMIP(const MIPInstance& mip)
{

    /* =========================
       OBJECTIVE
       ========================= */

    if (mip.objSense > 0)
        consoleLog("Minimize\n  obj: ");
    else
        consoleLog("Maximize\n  obj: ");

    bool first = true;

    if (!mip.obj_cols.empty())
    {
        for (size_t k = 0; k < mip.obj_cols.size(); ++k)
        {
            int j = mip.obj_cols[k];
            double coef = mip.obj_coefs[k];


            if (!first)
                consoleLogNoBreak(coef >= 0 ? " + " : " - ");
            else if (coef < 0)
                consoleLogNoBreak("-");

            const std::string var =
                mip.cNames.empty() ? "x" + std::to_string(j) : mip.cNames[j];

            consoleLogNoBreak("{} {}", std::fabs(coef), var);
            first = false;
        }
    }
    else
    {
        for (int j = 0; j < mip.ncols; ++j)
        {
            double coef = mip.obj[j];

            if (!first)
                consoleLogNoBreak(coef >= 0 ? " + " : " - ");
            else if (coef < 0)
                consoleLogNoBreak("-");

            const std::string var =
                mip.cNames.empty() ? "x" + std::to_string(j) : mip.cNames[j];

            consoleLogNoBreak("{} {}", std::fabs(coef), var);
            first = false;
        }
    }

        consoleLog(" + {}", mip.objOffset);

    consoleLog("\n\nSubject To\n");

    /* =========================
       CONSTRAINTS
       ========================= */
	auto beg = mip.rows.beg;
	auto ind = mip.rows.ind;
	auto val = mip.rows.val;
    for (int r = 0; r < mip.nrows; ++r) {
	    const std::string rname =
			mip.rNames.empty() ? "c" + std::to_string(r) : mip.rNames[r];

    	consoleLogNoBreak("  {}: ", rname);

    	bool firstTerm = true;


	    for (int k = beg[r]; k < beg[r + 1]; ++k) {
    		int j = ind[k];
    		double coef = val[k];

    		if (!firstTerm)
    			consoleLogNoBreak(coef >= 0 ? " + " : " - ");
    		else if (coef < 0)
    			consoleLogNoBreak("-");

    		const std::string var =
				mip.cNames.empty() ? "x" + std::to_string(j) : mip.cNames[j];

    		consoleLogNoBreak("{} {}", std::fabs(coef), var);
    		firstTerm = false;
    	}

    	if ( r <= mip.n_equalities ) {
    		consoleLogNoBreak(" = ");
    	}
    	else
    		consoleLogNoBreak(" <= ");

        consoleLog("{}\n", mip.rhs[r]);
    }

    /* =========================
       BOUNDS
       ========================= */

    consoleLog("\nBounds\n");

    for (int j = 0; j < mip.ncols; ++j)
    {
        const std::string cname =
            mip.cNames.empty() ? "x" + std::to_string(j) : mip.cNames[j];

        double lb = mip.lb[j];
        double ub = mip.ub[j];

        if (lb <= -1e20 && ub >= 1e20)
            consoleLog("  {} free\n", cname);
        else if (lb <= -1e20)
            consoleLog("  {} <= {}\n", cname, ub);
        else if (ub >= 1e20)
            consoleLog("  {} <= {}\n", lb, cname);
        else
            consoleLog("  {} <= {} <= {}\n", lb, cname, ub);
    }

    consoleLog("End");
}

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
	double timeLimit = 3000;
	int threads = 32;
	int maxTries = 1;

	/* Output. */
	bool enableOutput = true;
	int displayInterval = 500;

	/* Depth-first-search parameters. */
	bool propagate = true;
	bool propagate_objective = true;
	bool repair = false;
	bool repair_objective = false; /** Whether to do MIP walk on the objective constraint (+ cutoff). */
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

	/* Which partial solution to use for the partial sol strategies. -1 ~= 0. */
	int partial_sol{-1};

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

	/* Pool for potentially storing partial solutions/infeasible solutions for FPR. */
	SolutionPool partials;

	// relaxations
	MIPModelPtr lp; //< LP relaxation model

	/* The LP solution is computed in parallel. It might not be available yet. If it has become available, lp_solution_ready is set. */
	std::atomic<bool> lp_solution_ready{false};
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
