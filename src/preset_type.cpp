/*
 * @file preset_type.cpp
 * @brief PresetType type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include "preset_type.h"

#include "consolelog.h"
#include "str_utils.h"

#include <unordered_map>

static const std::unordered_map<std::string, PresetType> presetTypeMap = {
    {"random", PresetType::RANDOM},
    {"random2", PresetType::RANDOM2},
    {"badobj", PresetType::BADOBJ},
    {"badobjcl", PresetType::BADOBJCL},
    {"goodobj", PresetType::GOODOBJ},
    {"goodobjcl", PresetType::GOODOBJCL},
    {"locks", PresetType::LOCKS},
    {"locks2", PresetType::LOCKS2},
    {"cliques", PresetType::CLIQUES},
    {"cliques2", PresetType::CLIQUES2},
    {"zerocore", PresetType::ZEROCORE},
    {"zerolp", PresetType::ZEROLP},
    {"core", PresetType::CORE},
    {"lp", PresetType::LP}};

PresetType PresetTypeFromString(const std::string &str)
{
    std::string lowerStr = toLower(str); // Convert input to lowercase
    auto it = presetTypeMap.find(lowerStr);
    if (it != presetTypeMap.end())
    {
        return it->second;
    }
    else
    {
        consoleError("Unknown PresetType: {}", str);
        return PresetType::UNKNOWN;
    }
}

std::pair<RankerType, ValueChooserType> getRankerAndValueChooserFromPreset(PresetType preset)
{

    switch (preset)
    {
    case PresetType::RANDOM:
        return {RankerType::TYPECL, ValueChooserType::RANDOM};
    case PresetType::RANDOM2:
        return {RankerType::RANDOM, ValueChooserType::RANDOM};
    case PresetType::BADOBJ:
        return {RankerType::TYPE, ValueChooserType::BAD_OBJ};
    case PresetType::BADOBJCL:
        return {RankerType::TYPECL, ValueChooserType::BAD_OBJ};
    case PresetType::GOODOBJ:
        return {RankerType::TYPE, ValueChooserType::GOOD_OBJ};
    case PresetType::GOODOBJCL:
        return {RankerType::TYPECL, ValueChooserType::GOOD_OBJ};
    case PresetType::LOCKS:
        return {RankerType::LR, ValueChooserType::LOOSE};
    case PresetType::LOCKS2:
        return {RankerType::LOCKS, ValueChooserType::LOOSE};
    case PresetType::CLIQUES:
        return {RankerType::CLIQUES, ValueChooserType::UP};
    case PresetType::CLIQUES2:
        return {RankerType::CLIQUES2, ValueChooserType::UP};
    case PresetType::ZEROCORE:
        return {RankerType::TYPECL, ValueChooserType::RANDOM_LP};
    case PresetType::ZEROLP:
        return {RankerType::TYPECL, ValueChooserType::RANDOM_LP};
    case PresetType::CORE:
        return {RankerType::TYPECL, ValueChooserType::RANDOM_LP};
    case PresetType::LP:
        return {RankerType::TYPECL, ValueChooserType::RANDOM_LP};
    default:
    case PresetType::UNKNOWN:
        consoleError("Preset type is UNKNOWN!");
        exit(1);
    }
}

std::string toString(PresetType preset)
{
    switch (preset)
    {
    case PresetType::RANDOM:
        return "RANDOM";
    case PresetType::RANDOM2:
        return "RANDOM2";
    case PresetType::BADOBJ:
        return "BADOBJ";
    case PresetType::BADOBJCL:
        return "BADOBJCL";
    case PresetType::GOODOBJ:
        return "GOODOBJ";
    case PresetType::GOODOBJCL:
        return "GOODOBJCL";
    case PresetType::LOCKS:
        return "LOCKS";
    case PresetType::LOCKS2:
        return "LOCKS2";
    case PresetType::CLIQUES:
        return "CLIQUES";
    case PresetType::CLIQUES2:
        return "CLIQUES2";
    case PresetType::ZEROCORE:
        return "ZEROCORE";
    case PresetType::ZEROLP:
        return "ZEROLP";
    case PresetType::CORE:
        return "CORE";
    case PresetType::LP:
        return "LP";
    default:
    case PresetType::UNKNOWN:
        return "UNKNOWN";
    }
}

void printPresets()
{
    consoleInfo("Available presets are:");
    for (const auto preset : presetTypeMap)
    {
        const auto [ranker, chooser] = getRankerAndValueChooserFromPreset(preset.second);
        consoleInfo("{}: Ranker {}; ValueChooser {}", preset.first, toString(ranker), toString(chooser));
    }
}
