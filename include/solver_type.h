/**
 * @file solver_type.h
 * @brief Solver type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include <string>

enum class SolverType
{
    COPT,
    XPRESS,
    GUROBI,
    CPLEX,
    UNKNOWN,
};

/* Static methods for conversion. */
SolverType SolverTypeFromString(const std::string &str);

/* Get string  from SolverType. */
std::string toString(SolverType lpAlgoType);

void printSolverTypes();