/**
 * @file preset_type.h
 * @brief PresetType type
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include "value_chooser_type.h"
#include "ranker_type.h"

#include <string>

enum class PresetType
{
    UNKNOWN,
    RANDOM,    /** TYPECL + RANDOM */
    RANDOM2,   /** RANDOM + RANDOM */
    BADOBJ,    /** TYPE + BAD_OBJ */
    BADOBJCL,  /** TYPECL + BAD_OBJ */
    GOODOBJ,   /** TYPE + GOOD_OBJ */
    GOODOBJCL, /** TYPECL + GOOD_OBJ */
    LOCKS,     /** LR + LOOSE */
    LOCKS2,    /** LOCKS + LOOSE */
    CLIQUES,   /** CLIQUES + UP */
    CLIQUES2,  /** CLIQUES2 + UP */
    ZEROCORE,  /** TYPECL + ZEROCORE */
    ZEROLP,    /** TYPECL + ZEROLP */
    CORE,      /** TYPECL + CORE */
    LP,        /** TYPECL + LP */
};

PresetType PresetTypeFromString(const std::string &str);

/* Whether a certain value chooser needs an LP to function. */
std::pair<RankerType, ValueChooserType> getRankerAndValueChooserFromPreset(PresetType preset);

std::string toString(PresetType preset);

void printPresets();