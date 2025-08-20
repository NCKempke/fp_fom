/**
 * @file tool_assert.h
 * @brief Custom assert implementation
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include <cassert>

void assert_fail(const char* expr, const char* file, int line);

/* The customised assertion macros */
#ifndef NDEBUG
#define FP_ASSERT(expr) ((expr) ? (void)(0) : assert_fail(#expr, __FILE__, __LINE__))
#else
#define FP_ASSERT(expr) (void)(0)
#endif

#define FP_ASSERT_IFF(prop1, prop2)               (FP_ASSERT((prop1) == (prop2)))
#define FP_ASSERT_IF_THEN(antecedent, consequent) (FP_ASSERT(!(antecedent) || (consequent)))
