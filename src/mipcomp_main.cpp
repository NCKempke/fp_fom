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

#include "dfs.h"
#include "evolution_search.cuh"
#include "fpr_worker.h"
#include "gpu_data.cuh"
#include "linear_propagator.h"
#include "mip.h"
#include "MpsParser.hpp"
#include "queue_threadsafe.h"
#include "solution.h"
#include "strategies.h"
#include "table_propagators.h"
#include "thread_pool.h"
#include "tool_app.h"
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

/* Submit n fix-and-propagate workers continuously attempting to run some fix and propagate. Return's a vector of n_workers stop flags that can be used to cancel a worker. */
static std::vector<std::unique_ptr<std::atomic<bool>>> submit_fpr_workers(MIPData& mip_data, ThreadPool& thread_pool, const std::vector<std::pair<RankerType, ValueChooserType>>& strategies, std::atomic<size_t>& global_counter, const double deadline, int n_workers, const Params& params)
{
	std::vector<std::unique_ptr<std::atomic<bool>>> flags;
	flags.reserve(n_workers);

	consoleInfo("Submitting {} fpr cpu workers with deadline {}, current time is {}", n_workers, deadline, gStopWatch().elapsed());

	/* Run all combinations. */
	for (int i = 0; i < n_workers; ++i)
	{
		flags.push_back(std::make_unique<std::atomic<bool>>(false));
	    auto& flag_ref = *(flags.back()); /* reference to this worker’s flag; to not pass flags as a reference */

		thread_pool.enqueue([&, deadline]{
			fpr_worker(mip_data, strategies, global_counter, deadline, flag_ref, params);
		});
	}

	return flags;
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

		FP_ASSERT(!data.lp_solution_ready.load());
		data.lp_solution_ready.store(true);
	}


	void writeSolToFile(const MIPInstance &mip, const std::vector<double> &sol, double best_obj)
	{
		if (sol.empty())
		{
			consoleLog("No solution found");
			return;
		}

		std::string solFile = getProbName(args.input[0]) + ".sol";
		consoleLog("Writing feasible solution to {}...", solFile);
		std::ofstream out(solFile);
		out << fmt::format("=obj= {:.17g}", best_obj);
		for (int j = 0; j < mip.ncols; j++)
		{
			out << fmt::format("{} {:.17g}", mip.cNames[j], sol[j]) << "\n";
		}
		consoleLog("Done");
	}

	MIPModelPtr make_lp_solver(const MIPInstance& mip)
	{
		MIPModelPtr model;
		if (params.solver == SolverType::CPLEX)
		{
#if HAS_CPLEX
			model = std::make_shared<CPXModel>();
#else
			consoleError("CPLEX as solver not available.");
			return nullptr;
#endif
		}
		else if (params.solver == SolverType::XPRESS)
		{
#if HAS_XPRESS
			model = std::make_shared<XPRSModel>();
#else
			consoleError("XPRESS as solver not available.");
			return nullptr;
#endif
		}
		else if (params.solver == SolverType::COPT)
		{
#if HAS_COPT
			model = std::make_shared<COPTModel>();
#else
			consoleError("COPT as solver not available.");
			return nullptr;
#endif
		}
		else if (params.solver == SolverType::GUROBI)
		{
#if HAS_GUROBI
			model = std::make_shared<GUROBIModel>();
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

		pass_lp_to_solver(mip, model);

		return model;
	}

	void set_mipcomp_params () {
		params.threads = 8;
		params.timeLimit = 300;

		params.repair = true;
		params.mipPresolve = false;
		params.postsolve = false;
		params.writeSol = true;

		/* Solver settings. */
		params.solver = SolverType::GUROBI;
		params.presolver = SolverType::UNKNOWN;

		params.solveLp = true;
	}

	std::unique_ptr<MIPData> read_config_and_problem() {
		consoleLog("Reading the configuration.");
		params.readConfig();

		/* set mip competition default parameters */
		set_mipcomp_params();

		// log config
		consoleInfo("[config]");
		consoleLog("gitHash = {}", FP_GIT_HASH);
		consoleLog("probFile = {}", args.input[0]);
		consoleLog("strategy = {}", toString(params.ranker) + "_" + toString(params.valueChooser));
		consoleLog("solver = {}", toString(params.solver));
		params.logToConsole();
		consoleLog("");

		/* Read the model. */
		consoleLog("Reading the problem.");

		/* Read the problem; create an LP relaxation solver and build MIPData. */
		MIPInstance origMip = MpsParser::loadProblem(args.input[0]);
		MIPModelPtr model = make_lp_solver(origMip);

		return std::make_unique<MIPData>(std::move(origMip), model, rankerNeedsCliqueCover(params.ranker));
	}

	void exec() override {
		gStopWatch();

		/* First, read the parameters and setup the MIP competition settings. */
		// TODO: also initialize the GPU here; check the number of SMs etc.
		auto mip_data = read_config_and_problem();
		const MIPInstance &mip = mip_data->mip;

		// TODO: write input to timing file here.
		consoleInfo("Reading time = {}", gStopWatch().lap());
		consoleLog("");
		consoleLog("Problem:  #rows={} #cols={} #nnz={}", mip.nrows, mip.ncols, mip.rows.val.size());
		consoleLog("");

		FP_ASSERT(params.threads >= 1);

		/* We run this thread + params.threads - 1 threads. */
		ThreadPool thread_pool(params.threads - 1);
		std::atomic<size_t> fpr_counter(0);
		std::vector<std::unique_ptr<std::atomic<bool>>> worker_flags;

		/* Vector containing all FPR strategies we want to run repeatedly. The worker threads iterate this queue and, until timeout, try each strategy (changing the random seed each time/the lp solution can get updated). TODO: extend this! */
		const static std::vector<std::pair<RankerType, ValueChooserType>> fpr_queue_cpu = {
			{RankerType::FRAC, ValueChooserType::RANDOM_LP},
			{RankerType::DUALS, ValueChooserType::RANDOM_LP},
			{RankerType::RANDOM, ValueChooserType::RANDOM_LP},
			{RankerType::REDCOSTS, ValueChooserType::RANDOM_LP},
			{RankerType::TYPE, ValueChooserType::RANDOM_LP}
		};

		/* We run in parallel: The root LP using PDLP, 6 CPU fix-and-propagate threads, 1 thread running the GPU evolution search.
		 *
		 * We do not start a separate thread for the root LP runs. Rather, this is done on this thread. After the thread finished, the current thread starts checking + writing solutions and maybe adjusts the amount of workers.
		 */
		// TODO: run GPU FPR on another thread as well (and only 5 CPU fix-and-propagate threads then).

		// TODO: solve PDLP in parallel!
		/* Potentially solve the LP relaxation using PDLP; submit a thread for this!. In the threads, run LP free FPR variants as long as possible. */
		if (params.solveLp)
			solve_initial_lp(*mip_data);

		consoleInfo("LP time = {}", gStopWatch().lap());

		worker_flags = submit_fpr_workers(*mip_data, thread_pool, fpr_queue_cpu, fpr_counter, gStopWatch().elapsed() + 	300, 6, params);

		// TODO: start evolution search in parallel!

		/* Dedicate one process to running the GPU loop. The other processes run FPR-CPU for now. */
		GpuModel gpu_data(mip);

		consoleInfo("Running evo search");
		EvolutionSearch evo_search(mip, gpu_data);
		evo_search.run();

		// TODO: Communicate stop if the threads to not stop themselves!

		/* Tell everyone to stop. */
		consoleInfo("Setting stop flags");
		for (auto & flag : worker_flags) {
			(*flag).store(true);
		}

		thread_pool.wait();

		// TODO: make this thread write solutions to file in parallel.
		if (params.writeSol) {
			if (mip_data->solpool.hasFeas()) {
				consoleInfo("Writing best found solution");
				auto best_sol = mip_data->solpool.getIncumbent();

				writeSolToFile(mip_data->mip, best_sol.x, best_sol.objval);
			} else {
				consoleInfo("Did not find feasible solution; cannot write best sol");
			}
		}

		/* Final information displayed on command line. */
		mip_data->solpool.print();

		consoleInfo("[results]");
		consoleLog("found = {}", static_cast<int>(mip_data->solpool.hasFeas()));
		consoleLog("primalBound = {}", mip_data->solpool.primalBound());
		if (mip_data->solpool.hasSols())
			consoleLog("minAbsViol = {}", mip_data->solpool.minViolation());
		else
			consoleLog("minAbsViol = {}", 1000000.0);

		consoleLog("time = {}", gStopWatch().elapsed());
	}
};

int main(int argc, char const *argv[])
{
	MyApp theApp;
	theApp.parseArgsAndConfig(argc, argv);
	return theApp.run();
}
