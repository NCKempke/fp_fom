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

#include "mip.h"
#include "propagation.h"

// WalkSAT like implementation for MIP
class WalkMIP
{
public:
	WalkMIP(const MIPData &_data, const Params &_params, PropagationEngine &_engine);
	void walk();
	void oneOpt();

private:
	// data
	const MIPData &data;
	const Params &params;
	PropagationEngine &engine;
	// state
	std::mt19937_64 rndgen;
	// work
	std::vector<int> candidates;
	std::vector<double> score;
	std::vector<double> shifts;
	// helpers
	void applyFlip(int var);
	void evalFlip(int var, int pickedRow, bool &isCand, double &damage);
	void applyShift(int var, double delta);
	void evalShift(int var, double delta, int pickedRow, bool &isCand, double &damage);
};
