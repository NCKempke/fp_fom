/**
 * @file fp_main.cpp
 * @brief Main fail for PDLP based FPR
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include "mip.h"
#include "worker.h"
#include "dfs.h"
#include "strategies.h"
#include "linear_propagator.h"
#include "table_propagators.h"
#include "tool_app.h"
#include "thread_pool.h"
#include "version.h"

#include <consolelog.h>
#include <fileconfig.h>
#include <path.h>
#include <str_utils.h>
#include <map>
#include <mutex>
#include <thread>
#include <timer.h>

#include <iostream>
#include <fstream>

#if HAS_CPLEX
#include <cpxmodel.h>
#endif

#if HAS_COPT
#include <coptmodel.h>
#endif

#if HAS_GUROBI
#include <gurobimodel.h>
#endif

#if HAS_XPRESS
#include <xprsmodel.h>
#endif

static const unsigned int MIN_NUMBER_OF_INPUTS = 1;
std::mutex globalMutex;

static void runDFS(WorkerDataPtr worker, Params params, int trial)
{
	const MIPData &data{worker->mipdata};
	const MIPInstance &mip{data.mip};
	PropagationEngine &engine{worker->engine};
	SolutionPool &pool{worker->solpool};
	const Domain &domain = engine.getDomain();
	int n = mip.ncols;
	FP_ASSERT(n == domain.ncols());

	params.seed = params.seed + 117 * trial;
	if (params.maxNodes == -1)
		params.maxNodes = std::max(params.minNodes, n + 1);

	consoleLog("DFS {}-{} seed={}", toString(params.ranker), toString(params.valueChooser), params.seed);

	// Create ranker and value chooser
	RankerPtr ranker = makeRanker(params.ranker, params, data);
	ValuePtr chooser = makeValueChooser(params.valueChooser, params, data);
	FP_ASSERT(ranker);
	FP_ASSERT(chooser);

	// Use either old or new branching strategy.
	std::unique_ptr<DFSStrategy> strategy;

	if (params.useOldBranching)
	{
		strategy = std::make_unique<BranchSimple>(data);
		dynamic_cast<BranchSimple &>(*strategy).setup(domain, ranker, chooser);
	}
	else
	{
		strategy = std::make_unique<BranchNew>(data);
		dynamic_cast<BranchNew &>(*strategy).setup(domain, ranker, chooser);
	}

	Domain::iterator mark = engine.mark();
	dfsSearch(worker, params, *strategy);

	// final and longer repair if still infeasible
	if (!pool.hasFeas() && pool.hasSols())
	{
		// load solution into engine
		SolutionPtr sol = pool.getSols()[0];
		FP_ASSERT(sol);
		for (int j = 0; j < n; j++)
			engine.fix(j, sol->x[j]);
		double oldViol = engine.violation();
		FP_ASSERT(oldViol > 0.0);

		consoleLog("Final repair attempt");
		params.maxRepairSteps *= 5;
		WalkMIP repair(data, params, engine);
		repair.walk();
		double newViol = engine.violation();
		consoleLog("Final repair outcome: viol {} -> {}", oldViol, newViol);

		// add to pool if the new solution is 'better' w.r.t. to feasibility
		if (newViol < oldViol)
		{
			std::vector<double> x{domain.lbs().begin(), domain.lbs().end()};
			double objval = evalObj(mip, x);
			bool isFeas = isSolFeasible(mip, x);
			sol = makeFromSpan(mip, x, objval, isFeas, newViol);

			sol->timeFound = gStopWatch().getElapsed();
			const std::string strat_name = fmt::format("{}_{}", toString(params.ranker), toString(params.valueChooser));
			sol->foundBy = strat_name;
			pool.add(sol);
		}
	}

	engine.undo(mark);
}

static void runSingleHeuristicParallel(
	MIPData &data,
	WorkerDataManager &wManager,
	Params params)
{
	/* Set parameters according to run type and strategy */
	consoleLog("Runing {}-{} seed={}", toString(params.ranker), toString(params.valueChooser), params.seed);

	WorkerDataPtr worker = wManager.get();

	/*  Run the heuristic. */
	runDFS(worker, params, 0);

	/* merge local solution pool into global one */
	std::lock_guard lock{globalMutex};
	data.solpool.merge(worker->solpool);

	wManager.release(worker);
}

/* Run a portfolio of heuristics in parallel. */
static void runPortfolio(MIPData &data, const Params &params)
{
	SolutionPool &solpool{data.solpool};

	WorkerDataManager wManager{data};
	ThreadPool thpool(params.threads);

	/* We run the rankers frac, duals, random, redcosts, and type with random_lp for now. */
	const std::vector<RankerType> rankers = {RankerType::FRAC, RankerType::DUALS, RankerType::RANDOM, RankerType::REDCOSTS};
	const std::vector<ValueChooserType> valueChoosers = {ValueChooserType::RANDOM_LP};

	/* Run all combinations. */
	for (auto &ranker : rankers)
	{
		for (auto &valueChooser : valueChoosers)
		{
			Params params_copy = params;
			params_copy.ranker = ranker;
			params_copy.valueChooser = valueChooser;

			thpool.enqueue([&data, &wManager, params_copy]()
						   { runSingleHeuristicParallel(data, wManager, params_copy); });
		}
	}

	thpool.wait();

	consoleLog("");
	consoleInfo("Heuristics done after {}s [sols={} feas={}]",
				gStopWatch().getElapsed(),
				solpool.getSols().size(),
				solpool.hasFeas());
}

/* Run user specified heuristic. */
static void runSingleHeuristic(MIPData &data, const Params &params)
{
	SolutionPool &solpool{data.solpool};

	int tries = 0;
	while (tries < params.maxTries)
	{
		WorkerDataPtr worker = std::make_shared<WorkerData>(data);
		Domain::iterator mark = worker->engine.mark();

		worker->engine.undo(mark);

		/*  Run the heuristic. */
		runDFS(worker, params, tries++);

		/* merge local solution pool into global one */
		data.solpool.merge(worker->solpool);

		/* stop on Ctrl-C or time limit */
		if (UserBreak)
			break;
		if (gStopWatch().getElapsed() >= params.timeLimit)
			break;
	}

	consoleLog("tries = {}", tries);
}

class MyApp : public App
{
protected:
	// params
	Params params;

	// helpers
	bool checkUsage()
	{
		if (args.input.size() < MIN_NUMBER_OF_INPUTS || args.input[0] == "-h")
		{
			consoleLog("Usage: fp_main instance_file [options]");
			consoleLog("");
			consoleLog("Available options (case sensitive!) are:");
			params.printUsage();

			return false;
		}
		return true;
	}

	// construct LP solution if needed by our var/value strategies
	void solve_initial_lp(MIPData &data)
	{
		int n = data.mip.ncols;

		consoleLog("Solving initial LP relaxation");

		/* Should the objective be zeroed? */
		if (params.zeroObj)
		{
			consoleLog("Zero-ing out LP objective");

			std::vector<double> zeros(n, 0.0);
			std::vector<int> allIdx(n, 0);
			std::iota(allIdx.begin(), allIdx.end(), 0);

			data.lp->objcoefs(n, allIdx.data(), zeros.data());
			data.lp->objOffset(0.0);
		}

		consoleLog("Using {} {} to solve the LP relaxation.", toString(params.solver), toString(params.lpMethod));

		data.primals = solveLP(data.lp, params, true);

		data.duals.resize(data.lp->nrows());
		data.reduced_costs.resize(data.lp->ncols());

		data.lp->dual_sol(data.duals.data());
		data.lp->reduced_costs(data.reduced_costs.data());
	}

	MIPModelPtr presolve(MIPModelPtr model)
	{
		/* Presolve the model. */
		model->presolve();
		MIPModelPtr premodel = model->presolvedModel();

		consoleLog("Presolved Problem: #rows={} #cols={} #nnz={}", premodel->nrows(), premodel->ncols(), premodel->nnz());
		gStopWatch().stop();
		consoleInfo("Presolve time = {}", gStopWatch().getPartial());
		gStopWatch().start();

		/* convert the model to another solver IFF solver != presolver; this will get rid of the currently stored premodel. */
		if (params.solver != params.presolver)
		{
			consoleLog("Converting the problem to {}", toString(params.solver));
			MIPModelPtr converted_model = premodel.get()->convertTo(toString(params.solver));
			premodel = converted_model;
		}

		return premodel;
	}

	void writeSolToFile(const MIPInstance &origMip, const std::vector<double> &sol)
	{
		if (sol.empty())
		{
			consoleLog("No solution found");
			return;
		}

		std::string solFile = getProbName(args.input[0]) + ".sol";
		consoleLog("Writing feasible solution to {}...", solFile);
		std::ofstream out(solFile);
		for (int j = 0; j < origMip.ncols; j++)
		{
			out << fmt::format("{} {:.17g}", origMip.cNames[j], sol[j]) << "\n";
		}
		consoleLog("Done");
	}

	std::vector<double> postsolveBestSol(const MIPData &data, MIPModelPtr &model, const MIPInstance &origMip, bool was_presolved)
	{
		std::vector<double> postsolved_sol;
		SolutionPtr best_sol;

		if (data.solpool.hasFeas())
		{
			consoleLog("Postsolving incumbent");
			best_sol = data.solpool.getIncumbent();
			FP_ASSERT(best_sol);
		}
		else
			consoleInfo("No feasible solution available!");

		if (best_sol != NULL && was_presolved)
		{
			consoleLog("Time starting postsolve = {}", gStopWatch().getElapsed());
			postsolved_sol = model->postsolveSolution(best_sol->x);
			consoleLog("Time finished postsolve = {}", gStopWatch().getElapsed());

			// double check it is still feasible
			FP_ASSERT(postsolved_sol.size() == origMip.ncols);
			FP_ASSERT(isSolFeasible(origMip, postsolved_sol));
		}

		return postsolved_sol;
	}

	MIPModelPtr make_presolver()
	{
		FP_ASSERT(params.presolver == SolverType::CPLEX || params.presolver == SolverType::GUROBI);

		if (params.presolver == SolverType::CPLEX)
		{
#if HAS_CPLEX
			consoleLog("Creating CPLEX presolver.");
			consoleLog("");
			return std::make_shared<CPXModel>();
#else
			consoleError("CPLEX not available - presolve not possible.");
			return nullptr;
#endif
		}
		else if (params.presolver == SolverType::GUROBI)
		{
#if HAS_GUROBI
			consoleLog("Creating GUROBI presolver.");
			consoleLog("");
			return std::make_shared<GUROBIModel>();
#else
			consoleError("GUROBI not available - presolve not possible.");
			return nullptr;
#endif
		}
		else
		{
			consoleError("Presolve with {} not possible; aborting", toString(params.presolver));
			exit(1);
		}
	}

	MIPModelPtr make_solver()
	{
		if (params.solver == SolverType::CPLEX)
		{
#if HAS_CPLEX
			return std::make_shared<CPXModel>();
#else
			consoleError("CPLEX as solver not available.");
			return nullptr;
#endif
		}
		else if (params.solver == SolverType::XPRESS)
		{
#if HAS_XPRESS
			return std::make_shared<XPRSModel>();
#else
			consoleError("XPRESS as solver not available.");
			return nullptr;
#endif
		}
		else if (params.solver == SolverType::COPT)
		{
#if HAS_COPT
			/* CPX copt model to have presolve (as COPT does not offer any) */
			return std::make_shared<COPTModel>();
#else
			consoleError("COPT as solver not available.");
			return nullptr;
#endif
		}
		else if (params.solver == SolverType::GUROBI)
		{
#if HAS_GUROBI
			return std::make_shared<GUROBIModel>();
#else
			consoleError("COPT as solver not available.");
			return nullptr;
#endif
		}
		else
		{
			consoleError("Solver {} not available; aborting", toString(params.solver));
			exit(1);
		}
	}

	void run_single_strategy()
	{
	}

	void exec()
	{
		gStopWatch().start();

		// read params
		params.readConfig();

		// log config
		consoleInfo("[config]");
		consoleLog("gitHash = {}", FP_GIT_HASH);
		consoleLog("probFile = {}", args.input[0]);
		consoleLog("strategy = {}", toString(params.ranker) + "_" + toString(params.valueChooser));
		consoleLog("solver = {}", toString(params.solver));
		params.logToConsole();
		consoleLog("");

		MIPModelPtr model;

		/* Presolve and reading is done for all strategies together. */
		if (params.mipPresolve)
		{
			model = make_presolver();
		}
		else
		{
			model = make_solver();
		}

		/* Read the model. */
		consoleLog("Reading the problem.");

		/* Actually read the problem. */
		model->readModel(args.input[0]);

		gStopWatch().stop();
		auto reading_time = gStopWatch().getPartial();
		gStopWatch().start();

		consoleInfo("Reading time = {}", reading_time);
		consoleLog("");

		/* Extract original mip data. */
		MIPInstance origMip = extract(model); // TODO: move further down..

		MIPModelPtr premodel;
		bool hasPresolvedModel;

		/* Generate the presolved data. Either, this is the original data or we actually do presolve! */
		if (params.mipPresolve)
		{
			consoleLog("Presolving the problem.");
			consoleLog("Original Problem:  #rows={} #cols={} #nnz={}", model->nrows(), model->ncols(), model->nnz());
			/* In the case where there is no presolve to be done it might be that premodel will be model even after this call. */
			premodel = presolve(model);
			hasPresolvedModel = true;
		}
		else
		{
			consoleLog("Not presolving the problem.");
			consoleLog("Original Problem:  #rows={} #cols={} #nnz={}", model->nrows(), model->ncols(), model->nnz());

			/* no presolve to be done; clone the original model */
			premodel = model->clone();
			hasPresolvedModel = false;
		}

		/* Initialize the presolved MIP data. */
		const bool construct_cliquecover = rankerNeedsCliqueCover(params.ranker);
		MIPData data(premodel, construct_cliquecover);

		FP_ASSERT(premodel);
		consoleLog("");

		/* Create LP relaxation. */
		data.lp = premodel;
		data.lp->switchToLP();

		/* Initialized propagators and do one round of propagation. */
		const MIPInstance &mip = data.mip;
		PropagationEngine engine{data};
		engine.add(PropagatorPtr{new CliquesPropagator{data.cliquetable}});
		engine.add(PropagatorPtr{new ImplPropagator{data.impltable}});
		engine.add(PropagatorPtr{new LinearPropagator{data}});
		engine.init(mip.lb, mip.ub, mip.xtype);
		const Domain &domain = engine.getDomain();
		bool infeas = engine.propagate(true);
		FP_ASSERT(!infeas);

		/* Potentially solve the LP relaxation. */
		if (params.solveLp)
			solve_initial_lp(data);

		gStopWatch().stop();
		consoleInfo("LP time = {}", gStopWatch().getPartial());
		gStopWatch().start();

		if (params.runPortfolio)
		{
			consoleInfo("Running portfolio!");
			const auto enableOutputOld = params.enableOutput;
			/* Disable output for the time being to not pollute the command line. */
			params.enableOutput = false;

			runPortfolio(data, params);

			params.enableOutput = enableOutputOld;
		}
		else
		{
			consoleInfo("Running single heuristic!");
			runSingleHeuristic(data, params);
		}

		/* If we presolved using CPLEX, postsolve our solutions. */
		if (params.postsolve)
		{
			std::vector<double> best_sol = postsolveBestSol(data, model, origMip, hasPresolvedModel);
			writeSolToFile(origMip, best_sol);
		}

		consoleInfo("Printing the solpool");
		// Print solution pool
		data.solpool.print();

		consoleInfo("[results]");
		consoleLog("found = {}", (int)data.solpool.hasFeas());
		consoleLog("primalBound = {}", data.solpool.primalBound());
		if (data.solpool.hasSols())
			consoleLog("minAbsViol = {}", data.solpool.minViolation());
		else
			consoleLog("minAbsViol = {}", 1000000.0);

		gStopWatch().stop();
		consoleLog("time = {}", gStopWatch().getElapsed());
	}
};

int main(int argc, char const *argv[])
{
	MyApp theApp;
	theApp.parseArgsAndConfig(argc, argv);
	return theApp.run();
}
