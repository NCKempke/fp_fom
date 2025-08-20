/**
 * @brief Worker Data structures and methods
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 *
 * @date 2021
 *
 * Copyright 2021 Domenico Salvagnin
 */

#pragma once

#include "mip.h"
#include "propagation.h"
#include "table_propagators.h"
#include "linear_propagator.h"
#include <mipmodel.h>
#include <mutex>

/* Worker (i.e., thread-local) data */
struct WorkerData
{
public:
	WorkerData(const MIPData &_mipdata) : mipdata{_mipdata}, engine{mipdata}
	{
		// init propagation engine
		const MIPInstance &mip = mipdata.mip;
		engine.add(PropagatorPtr{new CliquesPropagator{mipdata.cliquetable}});
		engine.add(PropagatorPtr{new ImplPropagator{mipdata.impltable}});
		engine.add(PropagatorPtr{new LinearPropagator{mipdata}});
		engine.init(mip.lb, mip.ub, mip.xtype);
		// set sense in pool
		solpool.setObjSense(mip.objSense);
		// clone LP relaxation
		lp = mipdata.lp->clone();
	}
	// data
	const MIPData &mipdata;
	PropagationEngine engine;
	SolutionPool solpool;
	MIPModelPtr lp; //< LP relaxation
};

using WorkerDataPtr = std::shared_ptr<WorkerData>;

/* Manager for worker data instances (basically a thread-safe arena) */
struct WorkerDataManager
{
public:
	WorkerDataManager(const MIPData &_mipdata) : mipdata{_mipdata} {}
	WorkerDataPtr get()
	{
		std::lock_guard lock{m};
		if (unused.empty())
			return std::make_shared<WorkerData>(mipdata);
		else
		{
			WorkerDataPtr w = unused.back();
			unused.pop_back();
			return w;
		}
	}
	void release(WorkerDataPtr w)
	{
		std::lock_guard lock{m};
		unused.push_back(w);
	}

private:
	const MIPData &mipdata;
	std::vector<WorkerDataPtr> unused;
	std::mutex m;
};
