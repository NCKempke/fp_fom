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
    LR,
    TYPE,
    OBJ,
    INFER_OBJ,
    TYPECL,
    LOCKS,
    CLIQUES,
    CLIQUES2,
    RANDOM,
    REDCOSTS,
    DUALS,
    FRAC,
    DUALS_BREAK_FRAC,
    FRAC_BREAK_DUALS,
    FRAC_BREAK_REDCOSTS,
    REDCOSTS_BREAK_FRAC,
    UNKNOWN,
};

/* Static methods for conversion. */
RankerType RankerTypeFromString(const std::string &str);

/* Whether a certain ranker needs to solve an LP to function. */
bool rankerNeedsLpSolve(RankerType ranker);

/* Whether a certain ranker needs a clique cover. */
bool rankerNeedsCliqueCover(RankerType ranker);

/* Get string from RankerType. */
std::string toString(RankerType rankerType);

void printRankers();
