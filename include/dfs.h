/**
 * @file dfs.h
 * @brief Depth first serach
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 * @contributor Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2020-2025
 *
 * Copyright 2020 Domenico Salvagnin
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include "branch.h"
#include "mip.h"
#include "propagation.h"
#include "tool_app.h"
#include "strategies.h"
#include "walkmip.h"

#include <consolelog.h>
#include <timer.h>

#include <numeric>
#include <iostream>

/* Node data structure */
struct Node
{
public:
	Node(Branch b, Domain::iterator _tp, size_t _d) : branch{b}, trailp(_tp), depth(_d) {}
	Branch branch;
	Domain::iterator trailp;
	size_t depth;
};

/** Perform DFS on a given problem: customization of behaviour is provided via StrategyT */
template <typename StrategyT>
void dfsSearch(const MIPData& data, PropagationEngine& engine, SolutionPool& pool, MIPModelPtr lp, const Params &params, StrategyT &&strategy)
{
	const MIPInstance &mip{data.mip};
	const Domain &domain = engine.getDomain();
	int n = mip.ncols;
	int consecutiveInfeas = 0;
	int maxConsecutiveInfeas = (int)(params.maxConsecutiveInfeas * n);

	FP_ASSERT(lp);
	FP_ASSERT(n == domain.ncols());

	std::vector<Node> nodes;
	std::vector<int> allIdx(n);
	std::iota(allIdx.begin(), allIdx.end(), 0);

	WalkMIP repair(data.mip, params, engine);

	// push root node
	Domain::iterator start_mark = engine.mark();
	nodes.emplace_back(Branch{}, start_mark, 0);
	int nodecnt = 0;
	int lpSolved = 0;
	int numSolutions = 0;
	size_t maxDepth = 0;

	const std::string strat_name = fmt::format("{}_{}", toString(params.ranker), toString(params.valueChooser));

	if (params.enableOutput) {
		consoleLog("{}: Time starting DFS = {}", strat_name, gStopWatch().elapsed());
	}

	gStopWatch().lap();

	auto branch2str = [&](const Branch &br)
	{
		const char *sense;
		if (br.sense == 'B')
			sense = "=";
		else if (br.sense == 'L')
			sense = ">=";
		else
			sense = "<=";
		return fmt::format("{} {} {}", mip.cNames[br.index], sense, br.bound);
	};

	// DFS
	while (!nodes.empty())
	{
		// pop node
		Node node = nodes.back();
		nodes.pop_back();
		// backtrack
		engine.undo(node.trailp);
		const Branch &branch = node.branch;
		// engine.debugChecks();
		// apply branch
		if (branch.index != -1)
		{
			// consoleLog("Apply branching {}", branch2str(branch));
			// standard variable branch
			if (branch.sense == 'U')
			{
				bool nodeInfeas = engine.changeUpperBound(branch.index, branch.bound);
				FP_ASSERT(!nodeInfeas);
			}
			else if (branch.sense == 'L')
			{
				bool nodeInfeas = engine.changeLowerBound(branch.index, branch.bound);
				FP_ASSERT(!nodeInfeas);
			}
			else
			{
				bool nodeInfeas = engine.changeLowerBound(branch.index, branch.bound);
				FP_ASSERT(!nodeInfeas);
				nodeInfeas = engine.changeUpperBound(branch.index, branch.bound);
				FP_ASSERT(!nodeInfeas);
			}
		}

		nodecnt++;
		maxDepth = std::max(maxDepth, node.depth);

		bool nodeInfeas = (!engine.violatedRows().empty());

		// propagate & repair
		if (params.propagate)
			nodeInfeas = engine.propagate(false);

		if (params.repair)
		{
			repair.walk();
			nodeInfeas = (!engine.violatedRows().empty());
		}

		if (params.enableOutput && ((nodecnt % params.displayInterval) == 0))
			consoleLog("{}: {} nodes processed: depth={} violation={} elapsed={}", strat_name,
					   nodecnt, node.depth, engine.violation(), gStopWatch().elapsed());

		if (!engine.violatedRows().empty())
		{
			consecutiveInfeas++;
			if (params.backtrackOnInfeas)
				continue;
		}
		else
		{
			consecutiveInfeas = 0;
		}

		// branch (other customization point)
		const Domain &domain = engine.getDomain();
		std::vector<Branch> branches = strategy.branch(domain, nodeInfeas, branch);

		if (branches.empty())
		{
			if (params.enableOutput) {
				consoleLog("{}: {} nodes processed: depth={} violation={} elapsed={}", strat_name,
						   nodecnt, node.depth, engine.violation(), gStopWatch().elapsed());
			}

			/* End of dive */

			/* If there are continuous variables, we need to compute their values by solving a LP */
			if (data.nContinuous && !nodeInfeas)
			{
				// consoleLog("Leaf: solving LP");
				// copy bounds from domain to LP
				lp->lbs(allIdx.size(), allIdx.data(), domain.lbs().data());
				lp->ubs(allIdx.size(), allIdx.data(), domain.ubs().data());

				// solve LP
				if (params.enableOutput) {
					consoleLog("{}: Time starting LP solve = {}", strat_name, gStopWatch().elapsed());
				}
				lp->intParam(IntParam::Threads, params.threads);
				lp->logging(params.enableOutput);
				lp->dblParam(DblParam::TimeLimit, std::max(params.timeLimit - gStopWatch().elapsed(), 0.0));

				if (params.enableOutput) {
					consoleInfo("DFS time: {}", gStopWatch().lap());
				}
				/* This should be solved with higher precision. We should always use 1e-6 for the tolerances and 1e-8 (default) for the gap! */
				lp->lpopt(solverChar(params.lpMethodFinal), 1e-6, 1e-8);

				if (params.enableOutput) {
					consoleLog("{}: Time finished LP solve = {}", strat_name, gStopWatch().elapsed());
					consoleInfo("{}: LP time = {}", strat_name, gStopWatch().lap());
				}
				lpSolved++;

				if (lp->isPrimalFeas())
				{
					if (params.enableOutput) {
						consoleLog("{}: LP solved to optimality!", strat_name);
					}

					// looks like we have found a solution
					std::vector<double> x(n);
					lp->sol(x.data());
					// copy solution back to engine
					for (int j = 0; j < domain.ncols(); j++)
					{
						if (!equal(domain.lb(j), domain.ub(j), ABS_FEASTOL))
						{
							engine.fix(j, std::max(std::min(domain.ub(j), x[j]), domain.lb(j)));
						}
						FP_ASSERT(equal(domain.lb(j), domain.ub(j), ABS_FEASTOL));
					}
					// FP_ASSERT( engine.violatedRows().empty() );
					if (params.enableOutput) {
						consoleLog("{}: Objective {}", strat_name, evalObj(mip, x));
					}

					if (!engine.violatedRows().empty())
					{
						nodeInfeas = true;
					}
				}
				else
				{
					if (params.enableOutput) {
						consoleLog("{}: LP relaxation infeasible", strat_name);
					}
					nodeInfeas = true;
				}
			}

			/* Apply 1-opt if feasible */
			if (!nodeInfeas)
			{
				repair.oneOpt();
				FP_ASSERT(engine.violatedRows().empty());
			}
			else if (data.nContinuous)
			{
				// The solution got infeasible when solving the LP for the continuous variables
				// Hence, the engine does not yet have a complete solution. Make sure it does.
				for (int j = 0; j < domain.ncols(); j++)
				{
					if (!equal(domain.lb(j), domain.ub(j), ABS_FEASTOL))
					{
						engine.fix(j, domain.lb(j));
					}
					FP_ASSERT(equal(domain.lb(j), domain.ub(j), ABS_FEASTOL));
				}
			}

			/* Add solution to pool no matter what */
			std::vector<double> x{domain.lbs().begin(), domain.lbs().end()};
			double objval = evalObj(mip, x);
			bool isFeas = isSolFeasible(mip, x);
			auto sol_ptr = makeFromSpan(mip, x, objval, isFeas, engine.violation());
			sol_ptr->timeFound = gStopWatch().elapsed();
			sol_ptr->foundBy = strat_name;

			pool.add(std::move(sol_ptr));
			if (isFeas)
				numSolutions++;
		}
		else
		{
			// consoleLog("{}-way branch at depth {}", branches.size(), node.depth);
			// add them in reverse order (this is a stack after all)
			for (auto itr = branches.rbegin(); itr != branches.rend(); ++itr)
			{
				nodes.emplace_back(*itr, domain.mark(), node.depth + 1);
			}
		}

		// termination criteria
		if ((nodecnt >= params.maxNodes) ||
			(lpSolved >= params.maxLpSolved) ||
			(numSolutions >= params.maxSolutions) ||
			(consecutiveInfeas >= maxConsecutiveInfeas) ||
			(gStopWatch().elapsed() >= params.timeLimit) ||
			UserBreak)
		{
			if (params.enableOutput) {
				consoleLog("{}: Limits reached", strat_name);
			}
			break;
		}
	}
	if (nodes.empty() && params.enableOutput)
		consoleLog("{}: Nodes emtpy", strat_name);

	// cleanup
	engine.undo(start_mark);

	// make sure we restore the correct bounds on the LP object
	lp->lbs(allIdx.size(), allIdx.data(), mip.lb.data());
	lp->ubs(allIdx.size(), allIdx.data(), mip.ub.data());

	if (params.enableOutput && lpSolved != 1 && numSolutions != 1) {
		consoleInfo("DFS time: {}", gStopWatch().lap());
	}

	if (params.enableOutput) {
		consoleLog("{}: Time finish DFS = {}", strat_name, gStopWatch().elapsed());
	}
}

static void runDFS(const MIPData& data, PropagationEngine& engine, SolutionPool& pool, MIPModelPtr lp, const Params& params)
{
	const MIPInstance &mip{data.mip};
	const Domain &domain = engine.getDomain();
	int n = mip.ncols;

	FP_ASSERT(n == domain.ncols());
	FP_ASSERT(lp != nullptr);

	FP_ASSERT(params.maxNodes != -1);

	// consoleLog("DFS {}-{} seed={}", toString(params.ranker), toString(params.valueChooser), params.seed);

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
	dfsSearch(data, engine, pool, lp, params, *strategy);

	// final and longer repair if still infeasible
	if (!pool.hasFeas() && pool.hasSols())
	{
		// load solution into engine
		Solution sol = pool.getSol(0);
		for (int j = 0; j < n; j++)
			engine.fix(j, sol.x[j]);
		double oldViol = engine.violation();
		FP_ASSERT(oldViol > 0.0);

		if (params.enableOutput) {
			consoleLog("Final repair attempt");
		}
		const int final_repair_steps = params.maxRepairSteps * 5;
		WalkMIP repair(data.mip, params, engine);
		repair.set_max_steps(final_repair_steps);

		repair.walk();
		double newViol = engine.violation();
		if (params.enableOutput) {
			consoleLog("Final repair outcome: viol {} -> {}", oldViol, newViol);
		}

		// add to pool if the new solution is 'better' w.r.t. to feasibility
		if (newViol < oldViol)
		{

			std::vector<double> x{domain.lbs().begin(), domain.lbs().end()};
			double objval = evalObj(mip, x);
			bool isFeas = isSolFeasible(mip, x);
			auto sol_ptr = makeFromSpan(mip, x, objval, isFeas, newViol);

			sol_ptr->timeFound = gStopWatch().elapsed();
			const std::string strat_name = fmt::format("{}_{}", toString(params.ranker), toString(params.valueChooser));
			sol_ptr->foundBy = strat_name;
			pool.add(std::move(sol_ptr));
		}
	}

	engine.undo(mark);
}