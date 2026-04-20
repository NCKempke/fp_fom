/*
 * @file value_chooser_type.cpp
 * @brief ValueChooser type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include "value_chooser_type.h"

#include "consolelog.h"
#include "str_utils.h"

#include <unordered_map>

static const std::unordered_map<std::string, ValueChooserType> valueChooserTypeMap = {
    {"good_obj", ValueChooserType::GOOD_OBJ},
    {"bad_obj", ValueChooserType::BAD_OBJ},
    {"infer_obj", ValueChooserType::INFER_OBJ},
    {"random", ValueChooserType::RANDOM},
    {"loose", ValueChooserType::LOOSE},
    {"random_lp", ValueChooserType::RANDOM_LP},
    {"up", ValueChooserType::UP},
    {"down", ValueChooserType::DOWN},
    {"random_up_down", ValueChooserType::RANDOM_UP_DOWN},
    {"round_int", ValueChooserType::ROUND_INT},
    {"split", ValueChooserType::SPLIT}};

ValueChooserType ValueChooserTypeFromString(const std::string &str)
{
    std::string lowerStr = toLower(str); // Convert input to lowercase
    auto it = valueChooserTypeMap.find(lowerStr);
    if (it != valueChooserTypeMap.end())
    {
        return it->second;
    }
    else
    {
        consoleError("Unknown ValueChooserType: {}", str);
        return ValueChooserType::UNKNOWN;
    }
}

bool valueChooserNeedsLpSolve(ValueChooserType valueChooser)
{
    return (valueChooser == ValueChooserType::RANDOM_LP || valueChooser == ValueChooserType::ROUND_INT);
}

std::string toString(ValueChooserType valueChooser)
{
    switch (valueChooser)
    {
    case ValueChooserType::UNKNOWN:
        return "UNKNOWN";
    case ValueChooserType::GOOD_OBJ:
        return "GOOD_OBJ";
    case ValueChooserType::BAD_OBJ:
        return "BAD_OBJ";
    case ValueChooserType::INFER_OBJ:
        return "INFER_OBJ";
    case ValueChooserType::RANDOM:
        return "RANDOM";
    case ValueChooserType::LOOSE:
        return "LOOSE";
    case ValueChooserType::RANDOM_LP:
        return "RANDOM_LP";
    case ValueChooserType::UP:
        return "UP";
    case ValueChooserType::DOWN:
        return "DOWN";
    case ValueChooserType::RANDOM_UP_DOWN:
        return "RANDOM_UP_DOWN";
    case ValueChooserType::ROUND_INT:
        return "ROUND_INT";
    case ValueChooserType::SPLIT:
        return "SPLIT";
    default:
        return "UNKNOWN"; // Fallback case
    }
}

void printValueChoosers()
{
    consoleInfo("Available value choosers are:");
    for (const auto chooser : valueChooserTypeMap)
    {
        consoleInfo("{}", chooser.first);
    }
}
