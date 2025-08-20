/**
 * @file timer.h
 * @brief CPU Stopwatch for benchmarks
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 *
 * @date 2019
 *
 * Copyright 2019 Domenico Salvagnin
 */

#include <ctime>
#include <cstdlib>

#include "timer.h"

StopWatch &gStopWatch()
{
	static StopWatch theStopWatch;
	return theStopWatch;
}

std::string currentDateTime()
{
	time_t now = time(0);
	struct tm tstruct;
	char buf[100];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y-%m-%d %X (%z)", &tstruct);
	return buf;
}
