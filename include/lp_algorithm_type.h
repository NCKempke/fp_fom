/**
 * @file lp_algorithm_type.h
 * @brief LpAlgorithm type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include <string>

enum class LpAlgorithmType
{
    PRIMAL_SIMPLEX,
    DUAL_SIMPLEX,
    BARRIER,
    BARRIER_CROSSOVER,
    FIRST_ORDER_METHOD,
    UNKNOWN,
};

/* Static methods for conversion. */
LpAlgorithmType LpAlgorithmTypeFromString(const std::string &str);

/* Get string from LpAlgorithmType. */
std::string toString(LpAlgorithmType lpAlgoType);

// TODO: this should change.
/* Get char (for mipmodel.h) of LP method. */
char solverChar(LpAlgorithmType lpAlgoType);

/* Print all available LP methods. */
void printLpMethods();