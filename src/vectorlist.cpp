/**
 * @file vectorlist.cpp
 * @brief Vector List data structure
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 */

#include "vectorlist.h"

VectorList VectorList::transpose() const
{
	VectorList transposed;

	/* compute value counts */
	FP_ASSERT(maxVal > 0);
	std::vector<int> count(maxVal, 0);
	for (int i = 0; i < numVec; i++)
	{
		for (const auto &v : (*this)[i])
		{
			FP_ASSERT((v >= 0) && (v < maxVal));
			count[v]++;
		}
	}

	/* compute beg (for bucket sort) */
	transposed.beg.resize(maxVal + 1);
	transposed.beg[0] = 0;
	for (int v = 1; v <= maxVal; v++)
	{
		transposed.beg[v] = transposed.beg[v - 1] + count[v - 1];
	}
	FP_ASSERT(transposed.beg[maxVal] == nNonzeros());

	std::vector<size_t> start = transposed.beg;

	transposed.data.resize(nNonzeros());

	/* bucket sort */
	for (int i = 0; i < numVec; i++)
	{
		for (const auto &v : (*this)[i])
			transposed.data[start[v]++] = i;
	}

	/* some checks */
	for (int v = 0; v < maxVal; v++)
	{
		FP_ASSERT(start[v] == transposed.beg[v + 1]);
	}

	transposed.numVec = maxVal;
	transposed.maxVal = numVec;

	return transposed;
}
