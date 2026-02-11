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

static void fpr_worker(MIPData &mip_data, MIPModelPtr lp,
                       const std::vector<std::pair<RankerType, ValueChooserType> > &strategies,
                       std::atomic<size_t> &global_index, const double deadline, const std::atomic<bool> &should_stop,
                       const std::pair<RankerType, ValueChooserType>& fallback_strat,
                       Params params) {
    /* Initialize propagation engine and lp solver. */
    //TODO: AH @ NK the lp is always copied by value. Is it not sufficient to copy it once?
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

        /* Check whether the LP is ready yet. */
        if (!lp_is_ready) {
            lp_is_ready = mip_data.lp_solution_ready.load(std::memory_order_acquire);
        }

        /* Get a globally unique index to figure the next strategy to run on this thread. */
        const size_t idx = global_index.fetch_add(1, std::memory_order_relaxed);
        const auto [ranker, chooser] = strategies[idx % strategies.size()];

        const int n_partials = mip_data.partials.n_sols();

        const bool needs_lp = rankerNeedsLpSolve(ranker) || valueChooserNeedsLpSolve(chooser);
        const bool needs_partial_sol = rankerNeedsPartial(ranker) || valueChooserNeedsPartial(chooser);

        /*
         * Validate that the requested (ranker, valueChooser) strategy can be executed.
         * Fallback is selected if:
         *   - LP-based components are required but LP data is not available, or
         *   - Partial-solution-based components are required but no partial solutions exist.
         */
        if ((needs_lp && !lp_is_ready) || (needs_partial_sol && mip_data.partials.n_sols() == 0)) {
            params.ranker = fallback_strat.first;
            params.valueChooser = fallback_strat.second;
        }
        else {
            // All required data available → use requested strategy.
            params.ranker = ranker;
            params.valueChooser = chooser;
            assert(!needs_partial_sol || mip_data.partials.n_sols() > 0);

            if (needs_partial_sol) {
                assert(mip_data.partials.n_sols() > 0);

                /* For now, we pick idx % n_sols. */
                params.partial_sol = static_cast<int>(idx) % n_partials;
                assert(0 <= params.partial_sol && params.partial_sol < mip_data.partials.n_sols());
            }
        }

        /* Set new objective cutoff. */
        if (params.propagate_objective) {
            const double obj_cutoff = mip_data.solpool.get_obj_cutoff();
            state.engine.update_obj_cutoff(obj_cutoff);
        }

        /* Update the seed in case we do the same experiment twice. */
        params.seed = seed_orig + ith_run + idx;

        runDFS(mip_data, state.engine, mip_data.solpool, state.lp, params);
        ++ith_run;
    }

    consoleLog("stop requested");
}