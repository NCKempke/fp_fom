/**
 * @file walkmip.h
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

#pragma once

#include <cstdint>
#include <random>

#include "consolelog.h"
#include "mip.h"
#include "propagation.h"

// WalkSAT like implementation for MIP
class WalkMIP
{
public:
	WalkMIP(const MIPInstance &mip, const Params &params, PropagationEngine &_engine);

	void set_max_steps(int max_steps) {
		maxRepairSteps = max_steps;
	}

	void walk();
	void oneOpt();

	~WalkMIP()
	{
		// consoleLog("Walked {} times", n_walk);
	}

private:
	int n_walk{};
	// data
	const MIPInstance& mip;
	/* Parameters. */
	uint64_t seed;
	double randomWalkProbability;
	int maxRepairSteps;
	int maxRepairNonImprove;
	double timeLimit;
	bool repair_objective;

	PropagationEngine &engine;
	// state
	std::mt19937_64 rndgen;
	// work
	std::vector<int> candidates;
	std::vector<double> score;
	std::vector<double> shifts;
	// helpers
	void applyFlip(int var);
	void evalFlipRow(int row, double coef, double deltaX, int pickedRow, bool& isCand, double& damage);
	void evalFlip(int var, int pickedRow, bool &isCand, double &damage);

	void applyShift(int var, double delta);

	void evalShiftRow(int row, double coef, double delta, int pickedRow, bool &isCand, double &damage);
	void evalShift(int var, double delta, int pickedRow, bool &isCand, double &damage);
};
