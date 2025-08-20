/**
 * @file lp_algorithm_type.cpp
 * @brief LpAlgorithm type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include "lp_algorithm_type.h"

#include <unordered_map>

#include "consolelog.h"
#include "str_utils.h"

const std::unordered_map<std::string, LpAlgorithmType> lpAlgorithmTypeMap = {
    {"primal", LpAlgorithmType::PRIMAL_SIMPLEX},
    {"dual", LpAlgorithmType::DUAL_SIMPLEX},
    {"barrier", LpAlgorithmType::BARRIER},
    {"barrier_crossover", LpAlgorithmType::BARRIER_CROSSOVER},
    {"fom", LpAlgorithmType::FIRST_ORDER_METHOD},
};

/* Static methods for conversion. */
LpAlgorithmType LpAlgorithmTypeFromString(const std::string &str)
{
    std::string lowerStr = toLower(str); // Convert input to lowercase
    auto it = lpAlgorithmTypeMap.find(lowerStr);
    if (it != lpAlgorithmTypeMap.end())
    {
        return it->second;
    }
    else
    {
        consoleError("Unknown LpAlgorithmType: {}", str);
        return LpAlgorithmType::UNKNOWN;
    }
}

/* Get string from LpAlgorithmType. */
std::string toString(LpAlgorithmType lpAlgoType)
{
    switch (lpAlgoType)
    {
    case LpAlgorithmType::PRIMAL_SIMPLEX:
        return "PRIMAL_SIMPLEX";
    case LpAlgorithmType::DUAL_SIMPLEX:
        return "DUAL_SIMPLEX";
    case LpAlgorithmType::BARRIER:
        return "BARRIER";
    case LpAlgorithmType::BARRIER_CROSSOVER:
        return "BARRIER_CROSSOVER";
    case LpAlgorithmType::FIRST_ORDER_METHOD:
        return "FIRST_ORDER_METHOD";
    default:
        return "UNKNOWN";
    }
}

char solverChar(LpAlgorithmType lpAlgoType)
{
    switch (lpAlgoType)
    {
    case LpAlgorithmType::PRIMAL_SIMPLEX:
        return 'p';
    case LpAlgorithmType::DUAL_SIMPLEX:
        return 'd';
    case LpAlgorithmType::BARRIER:
        return 'b';
    case LpAlgorithmType::BARRIER_CROSSOVER:
        return 'c';
    case LpAlgorithmType::FIRST_ORDER_METHOD:
        return 'f';
    default:
        return 'x';
    }
}

void printLpMethods()
{
    consoleInfo("Available lp methods are:");
    for (const auto method : lpAlgorithmTypeMap)
    {
        consoleInfo("{}", method.first);
    }
}
