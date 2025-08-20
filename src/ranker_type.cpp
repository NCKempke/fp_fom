/**
 * @file ranker_type.cpp
 * @brief Ranker type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include "ranker_type.h"

#include <unordered_map>

#include "consolelog.h"
#include "str_utils.h"

const std::unordered_map<std::string, RankerType> rankerTypeMap = {
    {"type", RankerType::TYPE},
    {"locks", RankerType::LOCKS},
    {"random", RankerType::RANDOM},
    {"duals", RankerType::DUALS},
    {"redcosts", RankerType::REDCOSTS},
    {"frac", RankerType::FRAC}};

/* Static methods for conversion. */
RankerType RankerTypeFromString(const std::string &str)
{
    std::string lowerStr = toLower(str); // Convert input to lowercase
    auto it = rankerTypeMap.find(lowerStr);
    if (it != rankerTypeMap.end())
    {
        return it->second;
    }
    else
    {
        consoleError("Unknown RankerType: {}", str);
        return RankerType::UNKNOWN;
    }
}

bool rankerNeedsLpSolve(RankerType ranker)
{
    return (ranker == RankerType::REDCOSTS);
}

/* Get string from RankerType. */
std::string toString(RankerType rankerType)
{
    switch (rankerType)
    {
    case RankerType::TYPE:
        return "TYPE";
    case RankerType::LOCKS:
        return "LOCKS";
    case RankerType::RANDOM:
        return "RANDOM";
    case RankerType::REDCOSTS:
        return "REDCOSTS";
    case RankerType::DUALS:
        return "DUALS";
    case RankerType::FRAC:
        return "FRAC";
    default:
        return "UNKNOWN";
    }
}

void printRankers()
{
    consoleInfo("Available rankers are:");
    for (const auto ranker : rankerTypeMap)
    {
        consoleInfo("{}", ranker.first);
    }
}
