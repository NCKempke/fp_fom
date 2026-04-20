/**
 * @file value_chooser_type.h
 * @brief ValueChooser type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include <string>

enum class ValueChooserType
{
    UNKNOWN,
    GOOD_OBJ,
    BAD_OBJ,
    INFER_OBJ,
    RANDOM,
    LOOSE,
    RANDOM_LP, // Random rounding of LP relaxation.
    UP,
    DOWN,
    RANDOM_UP_DOWN,
    ROUND_INT,
    SPLIT,
};

ValueChooserType ValueChooserTypeFromString(const std::string &str);

/* Whether a certain value chooser needs an LP to function. */
bool valueChooserNeedsLpSolve(ValueChooserType valueChooser);

std::string toString(ValueChooserType valueChooser);

void printValueChoosers();