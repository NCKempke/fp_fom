/**
 * @file solver_type.cpp
 * @brief Solver type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include "solver_type.h"

#include <unordered_map>

#include "consolelog.h"
#include "str_utils.h"

const std::unordered_map<std::string, SolverType> solverTypeMap = {
    {"copt", SolverType::COPT},
    {"xpress", SolverType::XPRESS},
    {"gurobi", SolverType::GUROBI},
    {"cplex", SolverType::CPLEX},
};

/* Static methods for conversion. */
SolverType SolverTypeFromString(const std::string &str)
{
    std::string lowerStr = toLower(str); // Convert input to lowercase
    auto it = solverTypeMap.find(lowerStr);
    if (it != solverTypeMap.end())
    {
        return it->second;
    }
    else
    {
        consoleError("Unknown SolverType: {}", str);
        return SolverType::UNKNOWN;
    }
}

/* Get string from SolverType. */
std::string toString(SolverType lpAlgoType)
{
    switch (lpAlgoType)
    {
    case SolverType::COPT:
        return "COPT";
    case SolverType::XPRESS:
        return "XPRESS";
    case SolverType::GUROBI:
        return "GUROBI";
    case SolverType::CPLEX:
        return "CPLEX";
    default:
        return "UNKNOWN";
    }
}

void printSolverTypes()
{
    consoleInfo("Available solvers are:");
    for (const auto solver : solverTypeMap)
    {
        consoleInfo("{}", solver.first);
    }
}
