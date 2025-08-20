/**
 * @file ranker_type.h
 * @brief Ranker type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include <string>

enum class RankerType
{
    TYPE,
    LOCKS,
    RANDOM,
    REDCOSTS,
    DUALS,
    FRAC,
    UNKNOWN,
};

/* Static methods for conversion. */
RankerType RankerTypeFromString(const std::string &str);

/* Whether a certain ranker needs to solve an LP to function. */
bool rankerNeedsLpSolve(RankerType ranker);

/* Get string from RankerType. */
std::string toString(RankerType rankerType);

void printRankers();
