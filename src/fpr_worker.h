/**
 * @brief Worker for FPR.
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2026
 *
 * Copyright 2026 Nils-Christian Kempke
 */

#pragma once

#include "dfs.h"
#include "mip.h"
#include "propagation.h"
#include "table_propagators.h"
#include "linear_propagator.h"

#include <mipmodel.h>
#include <timer.h>

#include <atomic>
#include <thread>
#include <mutex>
#include <vector>

/* Local data of worker. This is mostly its own propagation engine. The worker also const references the mip problem and the LP solver (to check whether an LP solution is available). TODO: for now, each worker owns a full copy of the LP solver. */
struct WorkerFprState
{
public:
	WorkerFprState(const MIPData &mipdata_, MIPModelPtr lp_) : mipdata{mipdata_}, engine{mipdata_.mip}, lp(lp_)
	{
        /* Initialize this worker's propagation engine. */
        const MIPInstance &mip = mipdata.mip;
		engine.add(PropagatorPtr{new CliquesPropagator{mipdata.cliquetable}});
		engine.add(PropagatorPtr{new ImplPropagator{mipdata.impltable}});
		engine.add(PropagatorPtr{new LinearPropagator{mip}});
		engine.init(mip.lb, mip.ub, mip.xtype);
	}
	// data
	const MIPData &mipdata;
	PropagationEngine engine;

    /* LP solver */
	MIPModelPtr lp;
};

static void fpr_worker(MIPData& mip_data, MIPModelPtr lp, const std::vector<std::pair<RankerType, ValueChooserType>>& strategies, std::atomic<size_t>& global_index, const double deadline, std::atomic<bool>& should_stop, Params params) {
    /* Initialize propagation engine and lp solver. */
    WorkerFprState state(mip_data, lp);
    int ith_run = 0;
    bool lp_is_ready = false;

    uint64_t seed_orig = params.seed;

    params.threads = 1;
    params.enableOutput = false;
    if (params.maxNodes == -1)
        params.maxNodes = std::max(params.minNodes, state.mipdata.mip.ncols + 1);

    consoleLog("Starting thread\n");

    while (!should_stop.load(std::memory_order_relaxed)) {
        if (gStopWatch().elapsed() >= deadline) {
            consoleLog("Deadline hit at {} >= {}", gStopWatch().elapsed(), deadline);
            break;
        }

        /* Get a globally unique index to figure the next strategy to run on this thread. */
        size_t idx = global_index.fetch_add(1, std::memory_order_relaxed);
        const auto strat = strategies[idx % strategies.size()];

        /* Check whether the LP is ready yet. */
        if (!lp_is_ready) {
            lp_is_ready = mip_data.lp_solution_ready.load(std::memory_order_acquire);
        }

        /* Skip LP based solutions until the LP solution is available. */
        if ((rankerNeedsLpSolve(strat.first) || valueChooserNeedsLpSolve(strat.second)) && !lp_is_ready) {
            continue;
        }

        /* If ranker or value chooser need a partial solution, check whether one exists and pick one, else, continue. */
        if (rankerNeedsPartial(strat.first) || valueChooserNeedsPartial(strat.second)) {

            const int n_partials = mip_data.partials.n_sols();
            if (n_partials == 0)
                continue;

            /* For now, we pick idx % n_sols. */
            params.partial_sol = idx % n_partials;

            assert(0 <= params.partial_sol && params.partial_sol < mip_data.partials.n_sols());
        }

        /* Set new objective cutoff. */
        if (params.propagate_objective) {
            const double obj_cutoff = mip_data.solpool.get_obj_cutoff();
            state.engine.update_obj_cutoff(obj_cutoff);
        }

        /* Update the seed in case we do the same experiment twice. */
        params.seed = seed_orig + ith_run;
        params.ranker = strat.first;
        params.valueChooser = strat.second;

        runDFS(mip_data, state.engine, mip_data.solpool, state.lp, params);
        ++ith_run;
    }

    consoleLog("stop requested");
}