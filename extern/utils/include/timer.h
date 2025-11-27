/**
 * @file timer.h
 * @brief CPU Stopwatch for benchmarks
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 * @contributor Nils-Christian Kempke  <nilskempke at gmail dot com>
 *
 * @date 2019 - 2025
 *
 * Copyright 2019 Domenico Salvagnin
 * Copyright 2025 Nils-Christian Kempke
 */
#pragma once

#include "tool_assert.h"
#include <string>
#include <chrono>

/**
 * Benchmarking class
 * Simulates a simple stopwatch for wallclock time
 */

class StopWatch
{
public:
	/** default constructor; stopwatch starts immediately */
	StopWatch();

	/** restart the lap-timer and return the elapsed lap (diff to either creation or last call to lap) */
	double lap();

	/** stop the stopwatch and return the elapsed time */
	double elapsed() const;

private:
	const std::chrono::high_resolution_clock::time_point stopwatch_begin;
	std::chrono::high_resolution_clock::time_point lap_start;
};

inline StopWatch::StopWatch() : stopwatch_begin(std::chrono::high_resolution_clock::now()), lap_start{stopwatch_begin}
{
}

inline double StopWatch::lap()
{
	auto now = std::chrono::high_resolution_clock::now();
	auto old_lap = lap_start;
	lap_start = now;

	return std::chrono::duration<double>(now - old_lap).count();
}

inline double StopWatch::elapsed() const
{
	auto now = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double>(now - stopwatch_begin).count();
}

/**
 * Global Stopwatch
 */

StopWatch &gStopWatch();

/**
 * Get current date/time as std::string, format is YYYY-MM-DD.HH:mm:ss
 */
std::string currentDateTime();
