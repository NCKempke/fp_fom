/**
 * @file tool_assert.cpp
 * @brief Custom assert implementation
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2024-2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include <stdio.h>
#include <stdlib.h>

#include "tool_assert.h"

/* The function that handles assertion failures */
void assert_fail(const char *expr, const char *file, int line)
{
    printf("AssertFailed: %s:%d: %s\n", file, line, expr);
    fflush(stdout);
    abort();
}
