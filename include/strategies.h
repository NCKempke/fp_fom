/**
 * @file strategies.h
 * @brief FPR strategies
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 * @contributor Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2022-2025
 *
 * Copyright 2022 Domenico Salvagnin
 * Copyright 2025 Nils-Christian Kempke
 */

#pragma once

#include "dfs.h"
#include "mip.h"
#include "ranker_type.h"
#include "value_chooser_type.h"

#include "floats.h"

#include <ctime>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

/* Policy to sort variables */
class Ranker
{
public:
	Ranker(uint64_t _seed = 0) : seed{_seed}, unif{0.0, 1.0} { rndgen.seed(seed); };
	virtual ~Ranker() {};
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) = 0;

protected:
	bool randBool()
	{
		return unif(rndgen) < 0.5;
	}

	double randZeroOne()
	{
		return unif(rndgen);
	}

	uint64_t seed;

private:
	std::mt19937_64 rndgen;
	std::uniform_real_distribution<double> unif; /** Uniform 0-1 distribution. */
};

using RankerPtr = std::shared_ptr<Ranker>;

/* Policy to choose a value for a given variable */
class ValueChooser
{
public:
	ValueChooser() = default;
	ValueChooser(uint64_t _seed) : seed{_seed}, unif{0.0, 1.0} { rndgen.seed(seed); }
	virtual ~ValueChooser() {};
	virtual double operator()(const MIPData &data, const Domain &domain, int var) = 0;

protected:
	bool randBool()
	{
		return unif(rndgen) < 0.5;
	}

	double randZeroOne()
	{
		return unif(rndgen);
	}

	uint64_t seed;

private:
	std::mt19937_64 rndgen;
	std::uniform_real_distribution<double> unif; /** Uniform 0-1 distribution. */
};

using ValuePtr = std::shared_ptr<ValueChooser>;

/************* Rankers *************/

/** Take integers and binaries as is from the problem. */
class LR : public Ranker
{
public:
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));

		std::vector<int> discreteVariables(data.mip.ncols - data.nContinuous);
		int pos = 0;
		for (int j = 0; j < data.mip.ncols; j++)
		{
			if (data.mip.xtype[j] == 'B' || data.mip.xtype[j] == 'I')
				discreteVariables[pos++] = j;
		}
		FP_ASSERT(pos == static_cast<int>(discreteVariables.size()));
		return discreteVariables;
	}
};

class ByType : public Ranker
{
public:
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));
		// bucket sort by type
		std::vector<int> sorted(data.mip.ncols - data.nContinuous);
		int startBin = 0;
		int startInt = startBin + data.nBinaries;
		for (int j = 0; j < data.mip.ncols; j++)
		{
			if (data.mip.xtype[j] == 'B')
				sorted[startBin++] = j;
			else if (data.mip.xtype[j] == 'I')
				sorted[startInt++] = j;
		}
		FP_ASSERT(startBin == data.nBinaries);
		FP_ASSERT(startInt == (data.nBinaries + data.nIntegers));

		return sorted;
	}
};

class ByFrac : public Ranker
{
public:
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));
		// bucket sort by type
		std::vector<int> sorted(data.mip.ncols - data.nContinuous);
		int sorted_pos = 0;
		for (int j = 0; j < data.mip.ncols; j++)
		{
			if ((data.mip.xtype[j] == 'B' || data.mip.xtype[j] == 'I'))
			{
				sorted[sorted_pos] = j;
				++sorted_pos;
			}
		}
		FP_ASSERT(sorted_pos == data.mip.ncols - data.nContinuous);

		/* Sort them by their reduced costs. */
		std::sort(sorted.begin(), sorted.end(), [&](int i, int j)
				  {
			/* x_i comes before x_j if frac(x_i) < frac(x_j); in this case, return true */
			return std::abs(std::round(data.primals[i]) - data.primals[i]) < std::abs(std::round(data.primals[j]) - data.primals[j]); });

		return sorted;
	}
};

class ByDuals : public Ranker
{
	/* The dual ranker does the following:
	 *
	 * 1) sort constraints by dual value, high duals indicate a strong influence of the constraint on the objective function.
	 * 2) Within each constraint, we sort the variables by reduced costs
	 */
public:
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		std::vector<int> rowsSortedByDual(data.mip.nrows);
		std::vector<int> colsSorted(data.mip.ncols - data.nContinuous);

		FP_ASSERT(data.duals.size() >= data.mip.nrows);

		std::iota(rowsSortedByDual.begin(), rowsSortedByDual.end(), 0);

		/* Sort by dual values. */
		std::sort(rowsSortedByDual.begin(), rowsSortedByDual.end(), [&](int i, int j)
				  {
			/* i before j if i's dual value is greater than j's */
			return std::fabs(data.duals[i]) > std::fabs(data.duals[j]); });

		/* Sort the rows by reduced costs. We iterate each row, extract the non-marked integers and binaries, sort them, and add them to the sorted list (+mark them). */
		std::vector<int> sorted(data.mip.ncols - data.nContinuous);
		/* Buffer for extracting the integers and binaries of a row. */
		std::vector<int> rowIntegerVariables(data.mip.ncols);
		/* Indicator whether we have seen a column already. */
		std::vector<bool> marked(data.mip.ncols, false);
		const int nIntBinCols = data.nBinaries + data.nIntegers;
		int nSorted = 0;

		const auto &matrix = data.mip.rows;
		const auto &rowBeg = matrix.beg;
		const auto &rowInd = matrix.ind;

		FP_ASSERT(data.reduced_costs.size() >= data.mip.ncols);
		FP_ASSERT(rowBeg.size() == data.mip.nrows);

		for (int iRow = 0; iRow < data.mip.nrows; ++iRow)
		{
			int nBinInt = 0;
			/* Extract the unmarked binaries and integers. */
			for (int iNz = rowBeg[iRow]; iNz < rowBeg[iRow + 1]; ++iNz)
			{
				const int jCol = rowInd[iNz];
				FP_ASSERT(0 <= jCol && jCol < data.mip.ncols);

				/* Skip non-binary, non-integer variables. */
				if (data.mip.xtype[jCol] == 'C')
					continue;

				if (marked[jCol])
					continue;
				else
					marked[jCol] = true;

				rowIntegerVariables[nBinInt] = jCol;
				++nBinInt;
			}

			if (nBinInt == 0)
				continue;

			/* Sort the extracted variables by reduced costs. */
			std::sort(rowIntegerVariables.begin(), rowIntegerVariables.begin() + nBinInt, [&](int i, int j)
					  {
				/* i before j if i's reduced costs are greater than j's */
				return fabs(data.reduced_costs[i]) > fabs(data.reduced_costs[j]); });

			/* Move the sorted variables to the sorted array. */
			std::copy(rowIntegerVariables.begin(), rowIntegerVariables.begin() + nBinInt, sorted.begin() + nSorted);
			nSorted += nBinInt;

			FP_ASSERT(nSorted <= nIntBinCols);

			if (nSorted == nIntBinCols)
				break;
		}

		FP_ASSERT(nSorted == nIntBinCols);

		return sorted;
	}
};

class DualsWithFracTieBreaker : public Ranker
{
public:
    virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
    {
        std::vector<int> sorted(data.mip.ncols - data.nContinuous);
        std::vector<int> rowIntegerVariables(data.mip.ncols);
        std::vector<bool> marked(data.mip.ncols, false);
        const int nIntBinCols = data.nBinaries + data.nIntegers;
        int nSorted = 0;

        const auto &matrix = data.mip.rows;
        const auto &rowBeg = matrix.beg;
        const auto &rowInd = matrix.ind;

        FP_ASSERT(data.duals.size() >= data.mip.nrows);

        std::vector<int> rowsSortedByDual(data.mip.nrows);
        std::iota(rowsSortedByDual.begin(), rowsSortedByDual.end(), 0);

        // Sort rows by dual value
        std::sort(rowsSortedByDual.begin(), rowsSortedByDual.end(), [&](int i, int j)
        {
            return std::fabs(data.duals[i]) > std::fabs(data.duals[j]);
        });

        for (int i = 0; i < data.mip.nrows; ++i)
        {
            int iRow = rowsSortedByDual[i];
            int nBinInt = 0;

            // Extract unmarked integer/binary variables
            for (int iNz = rowBeg[iRow]; iNz < rowBeg[iRow + 1]; ++iNz)
            {
                int jCol = rowInd[iNz];
                if (data.mip.xtype[jCol] == 'C' || marked[jCol])
                    continue;
                marked[jCol] = true;
                rowIntegerVariables[nBinInt++] = jCol;
            }

            if (nBinInt == 0)
                continue;

            // Sort extracted variables: primary by reduced cost (optional) but tie-break by fraction
            std::sort(rowIntegerVariables.begin(), rowIntegerVariables.begin() + nBinInt, [&](int i, int j)
            {
                double frac_i = std::abs(std::round(data.primals[i]) - data.primals[i]);
                double frac_j = std::abs(std::round(data.primals[j]) - data.primals[j]);

                if (std::fabs(data.reduced_costs[i] - data.reduced_costs[j]) > 1e-12)
                    return std::fabs(data.reduced_costs[i]) > std::fabs(data.reduced_costs[j]); // maintain original duals ordering
                else
                    return frac_i < frac_j; // tie-break by fraction
            });

            std::copy(rowIntegerVariables.begin(), rowIntegerVariables.begin() + nBinInt, sorted.begin() + nSorted);
            nSorted += nBinInt;

            if (nSorted == nIntBinCols)
                break;
        }

        FP_ASSERT(nSorted == nIntBinCols);
        return sorted;
    }
};

class FracWithDualsTieBreaker : public Ranker
{
public:
    virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
    {
        std::vector<int> sorted(data.mip.ncols - data.nContinuous);
        size_t sorted_pos = 0;

        for (int j = 0; j < data.mip.ncols; ++j)
        {
            if (data.mip.xtype[j] == 'B' || data.mip.xtype[j] == 'I')
                sorted[sorted_pos++] = j;
        }

        FP_ASSERT(sorted_pos == data.mip.ncols - data.nContinuous);

        std::sort(sorted.begin(), sorted.end(), [&](int i, int j)
        {
            double frac_i = std::abs(std::round(data.primals[i]) - data.primals[i]);
            double frac_j = std::abs(std::round(data.primals[j]) - data.primals[j]);

            if (frac_i != frac_j)
                return frac_i < frac_j; // primary: fractionality
            else
                return std::fabs(data.duals[i]) > std::fabs(data.duals[j]); // tie-break: duals
        });

        return sorted;
    }
};


class ByReducedCosts : public Ranker
{
public:
	ByReducedCosts(uint64_t seed) : Ranker{seed} {}

	std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		assert(data.reduced_costs.size() >= data.mip.ncols);

		/* Get all integral variables. */
		size_t sorted_pos = 0;
		std::vector<int> sorted(data.mip.ncols - data.nContinuous);
		for (int j = 0; j < data.mip.ncols; j++)
		{
			if ((data.mip.xtype[j] == 'B' || data.mip.xtype[j] == 'I'))
			{
				sorted[sorted_pos] = j;
				++sorted_pos;
			}
		}

		FP_ASSERT(sorted_pos == data.mip.ncols - data.nContinuous);

		/* Sort them by their reduced costs. */
		std::sort(sorted.begin(), sorted.end(), [&](int i, int j)
				  {
			/* x_i comes before x_j if [u_i - l_i] * red_cost_i > [u_j - l_j] * red_cost_j; in this case, return true */
			return data.reduced_costs[i] > data.reduced_costs[j]; });
		return sorted;
	}
};

class FracWithRedCostTieBreaker : public Ranker
{
public:
    virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
    {
        FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));

        std::vector<int> sorted(data.mip.ncols - data.nContinuous);
        int sorted_pos = 0;
        for (int j = 0; j < data.mip.ncols; j++)
        {
            if (data.mip.xtype[j] == 'B' || data.mip.xtype[j] == 'I')
            {
                sorted[sorted_pos] = j;
                ++sorted_pos;
            }
        }
        FP_ASSERT(sorted_pos == data.mip.ncols - data.nContinuous);

        std::sort(sorted.begin(), sorted.end(), [&](int i, int j)
        {
            double frac_i = std::abs(std::round(data.primals[i]) - data.primals[i]);
            double frac_j = std::abs(std::round(data.primals[j]) - data.primals[j]);

            if (frac_i != frac_j)
                return frac_i < frac_j; // primary sort by fraction
            else
                return data.reduced_costs[i] > data.reduced_costs[j]; // tie-break by reduced cost
        });

        return sorted;
    }
};

class RedCostWithFracTieBreaker : public Ranker
{
public:
    RedCostWithFracTieBreaker(uint64_t seed) : Ranker{seed} {}

    std::vector<int> operator()(const MIPData &data, const Domain &domain) override
    {
        assert(data.reduced_costs.size() >= data.mip.ncols);

        std::vector<int> sorted(data.mip.ncols - data.nContinuous);
        size_t sorted_pos = 0;
        for (int j = 0; j < data.mip.ncols; j++)
        {
            if (data.mip.xtype[j] == 'B' || data.mip.xtype[j] == 'I')
            {
                sorted[sorted_pos] = j;
                ++sorted_pos;
            }
        }

        FP_ASSERT(sorted_pos == data.mip.ncols - data.nContinuous);

        std::sort(sorted.begin(), sorted.end(), [&](int i, int j)
        {
            if (data.reduced_costs[i] != data.reduced_costs[j])
                return data.reduced_costs[i] > data.reduced_costs[j]; // primary sort by reduced cost
            else
            {
                double frac_i = std::abs(std::round(data.primals[i]) - data.primals[i]);
                double frac_j = std::abs(std::round(data.primals[j]) - data.primals[j]);
                return frac_i < frac_j; // tie-break by fraction
            }
        });

        return sorted;
    }
};

class RandomOrder : public Ranker
{
public:
	RandomOrder(uint64_t seed) : Ranker{seed} {}
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		// bucket sort by type
		std::vector<int> sorted(data.mip.ncols - data.nContinuous);
		int startBin = 0;
		int startInt = startBin + data.nBinaries;
		for (int j = 0; j < data.mip.ncols; j++)
		{
			if (data.mip.xtype[j] == 'B')
				sorted[startBin++] = j;
			else if (data.mip.xtype[j] == 'I')
				sorted[startInt++] = j;
		}
		FP_ASSERT(startBin == data.nBinaries);
		FP_ASSERT(startInt == (data.nBinaries + data.nIntegers));
		std::mt19937_64 rndgen;
		rndgen.seed(seed);
		std::shuffle(sorted.begin(), sorted.begin() + data.nBinaries, rndgen);
		std::shuffle(sorted.begin() + data.nBinaries, sorted.end(), rndgen);
		return sorted;
	}
};

class Locks : public Ranker
{
public:
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		// compute score
		std::vector<int> score(data.mip.ncols);
		int sentinel = 2 * data.mip.nrows;
		for (int var = 0; var < data.mip.ncols; var++)
		{
			if (data.mip.xtype[var] == 'C')
			{
				// continuous are the very last and will be purged from the list
				score[var] = sentinel;
				continue;
			}
			// both bounds are finite
			score[var] = data.uplocks[var];
			if (data.uplocks[var] > data.dnlocks[var])
				score[var] = -data.dnlocks[var];
			else
				score[var] = -data.uplocks[var];
		}

		// sort by increasing score
		std::vector<int> sorted(data.mip.ncols);
		std::iota(sorted.begin(), sorted.end(), 0);
		std::sort(sorted.begin(), sorted.end(), [&](int v1, int v2)
				  { return (score[v1] < score[v2]); });

		// drop continuous if any
		if (data.nContinuous)
		{
			int start = data.mip.ncols - data.nContinuous;
			FP_ASSERT(std::all_of(sorted.begin() + start, sorted.end(), [&](int v)
								  { return (score[v] == sentinel); }));
			sorted.erase(sorted.begin() + start, sorted.end());
		}

		return sorted;
	}
};

class ByTypeCl : public Ranker
{
public:
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		/* Add all variables w.r.t. their clique cover. Then place integers. */
		FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));
		// bucket sort by type
		std::vector<int> sorted(data.mip.ncols - data.nContinuous);
		std::vector<bool> added(data.mip.ncols);

		int startBin = 0;
		int startInt = startBin + data.nBinaries;

		/* Iterate the clique-cover and put all covered binaries. Then, put the uncovered binaries. */
		for (int cl = 0; cl < data.cliquecover.nCliques(); cl++)
		{
			std::vector<int> vars;
			std::vector<double> weights;
			for (int lit : data.cliquecover.getClique(cl))
			{
				const auto [j, isPos] = varFromLit(lit, data.mip.ncols);

				if (!added[j])
				{
					added[j] = true;
					sorted[startBin++] = j;
				}
			}
		}

		/* Put uncovered binaries. Simultaneously, put integers last. */
		for (int j = 0; j < data.mip.ncols; j++)
		{
			if ((domain.type(j) == 'B') && !added[j])
			{
				added[j] = true;
				sorted[startBin++] = j;
			}
			if (data.mip.xtype[j] == 'I')
			{
				added[j] = true;
				sorted[startInt++] = j;
			}
		}
		FP_ASSERT(startBin == data.nBinaries);
		FP_ASSERT(startInt == (data.nBinaries + data.nIntegers));

		return sorted;
	}
};

class ByCliques : public Ranker
{
public:
	ByCliques(uint64_t seed) : Ranker{seed} {}
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		std::vector<int> sorted;
		std::vector<bool> added(data.mip.ncols, false);

		for (int cl = 0; cl < data.cliquecover.nCliques(); cl++)
		{
			std::vector<int> vars;
			std::vector<double> weights;
			for (int lit : data.cliquecover.getClique(cl))
			{
				const auto [j, isPos] = varFromLit(lit, data.mip.ncols);

				if (equal(domain.lb(j), domain.ub(j)))
					continue;

				vars.push_back(j);

				if (isPos)
					weights.push_back(data.primals[j]);
				else
					weights.push_back(1.0 - data.primals[j]);
			}

			for (size_t inz = 0; inz < weights.size(); ++inz)
			{
				weights[inz] = std::log(randZeroOne() / weights[inz]);
			}

			// Create an index array to sort by weights
			std::vector<size_t> order(weights.size());
			std::iota(order.begin(), order.end(), 0);

			// Sort the indices based on corresponding weights (non-decreasing)
			std::sort(order.begin(), order.end(),
					  [&](size_t a, size_t b)
					  { return weights[a] < weights[b]; });

			// Reorder vars accordingly (no need to reorder weights)
			std::vector<int> sorted_vars(vars.size());
			for (size_t i = 0; i < order.size(); ++i)
				sorted_vars[i] = vars[order[i]];

			// add them to sorted list
			for (auto j : sorted)
			{
				if (!added[j])
				{
					added[j] = true;
					sorted.push_back(j);
				}
			}
		}

		for (int j = 0; j < data.mip.ncols; j++)
		{
			if ((domain.type(j) != 'C') && !added[j])
				sorted.push_back(j);
		}

		assert(static_cast<int>(sorted.size()) == data.mip.ncols - data.nContinuous);

		return sorted;
	}
};

class ByCliques2 : public Ranker
{
public:
	ByCliques2(uint64_t seed) : Ranker{seed} {}
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		std::vector<int> sorted;
		std::vector<bool> added(data.mip.ncols, false);

		for (int cl = 0; cl < data.cliquetable.nCliques(); cl++)
		{
			double bestValue = 0.0;
			double sum = 0.0;
			int bestVar = -1;
			bool skip_clique = false;

			for (int lit : data.cliquetable.getClique(cl))
			{
				const auto [j, isPos] = varFromLit(lit, data.mip.ncols);

				if (isPos)
				{
					if (equal(domain.lb(j), 1.0))
					{
						skip_clique = true;
						break;
					}

					const double v = data.primals[j];
					sum += v;

					if (v > bestValue && equal(domain.ub(j), 1.0))
					{
						bestVar = j;
						bestValue = v;
					}
				}
				else
				{
					if (equal(domain.ub(j), 0.0))
					{
						skip_clique = true;
						break;
					}

					const double v = (1.0 - data.primals[j]);
					sum += v;

					if (v > bestValue && equal(domain.lb(j), 0.0))
					{
						bestVar = j;
						bestValue = v;
					}
				}
			}

			if (!skip_clique && bestVar != -1 && equal(sum, 1.0))
			{
				if (!added[bestVar])
				{
					sorted.push_back(bestVar);
					added[bestVar] = true;
				}

				for (int lit : data.cliquetable.getClique(cl))
				{
					const auto [j, isPos] = varFromLit(lit, data.mip.ncols);

					/* This check also skips bestVar. */
					if (!added[j])
					{
						sorted.push_back(j);
						added[j] = true;
					}
				}
			}
		}

		for (int j = 0; j < data.mip.ncols; j++)
		{
			if ((domain.type(j) != 'C') && !added[j])
				sorted.push_back(j);
		}

		assert(static_cast<int>(sorted.size()) == data.mip.ncols - data.nContinuous);

		return sorted;
	}
};

/************* Value Choosers *************/
class GoodObj : public ValueChooser
{
public:
	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		return (data.mip.objSense * data.mip.obj[var] >= 0.0) ? domain.lb(var) : domain.ub(var);
	}
};

class BadObj : public ValueChooser
{
public:
	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		return (data.mip.objSense * data.mip.obj[var] <= 0.0) ? domain.lb(var) : domain.ub(var);
	}
};

class Loose : public ValueChooser
{
public:
	double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		if (data.uplocks[var] > data.dnlocks[var])
			return domain.lb(var);
		else if (data.uplocks[var] < data.dnlocks[var])
			return domain.ub(var);
		// in case of ties consider objective, again in the loose direction
		return (data.mip.objSense * data.mip.obj[var] <= 0.0) ? domain.lb(var) : domain.ub(var);
	}
};

class RoundInt : public ValueChooser
{
public:
	RoundInt(const std::vector<double> &_xref) : xref(_xref) {};
	double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		FP_ASSERT(!xref.empty());
		if (greaterEqualThan(xref[var], domain.ub(var), ABS_FEASTOL))
			return domain.ub(var);
		if (lessEqualThan(xref[var], domain.lb(var), ABS_FEASTOL))
			return domain.lb(var);
		double fp = fractionalPart(xref[var]);
		if (fp >= 0.5)
			return ceilEps(xref[var]);
		else
			return floorEps(xref[var]);
	}

protected:
	std::vector<double> xref;
};

class AlwaysUp : public ValueChooser
{
public:
	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		return domain.ub(var);
	}
};

class AlwaysDown : public ValueChooser
{
public:
	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		return domain.lb(var);
	}
};

class Split : public ValueChooser
{
public:
	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		return std::floor((domain.ub(var) + domain.lb(var)) / 2.0);
	}
};

class RandomUpDown : public ValueChooser
{
public:
	RandomUpDown(uint64_t seed) : ValueChooser{seed} {};

	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		if (randBool())
		{
			return domain.lb(var);
		}
		else
		{
			return domain.ub(var);
		}
	}
};

class RandomValue : public ValueChooser
{
public:
	RandomValue(uint64_t seed) : ValueChooser{seed} {};

	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		double lb = domain.lb(var);
		double ub = domain.ub(var);
		if (domain.type(var) == 'B')
		{
			if (randBool())
				return lb;
			else
				return ub;
		}
		else
		{
			FP_ASSERT(domain.type(var) == 'I');
			double value = floorEps(lb + randZeroOne() * (ub - lb) + 0.5);
			value = std::min(value, ub);
			value = std::max(value, lb);
			return value;
		}
	}
};

class RandomRelaxation : public ValueChooser
{
public:
	RandomRelaxation(uint64_t seed, const std::vector<double> &x)
		: ValueChooser{seed}, xref{x} {};

	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		const double lb = domain.lb(var);
		const double ub = domain.ub(var);
		double fixingVal = 0.0;
		double randVal = randZeroOne();

		if (domain.type(var) == 'B')
		{
			if (randVal >= xref[var])
				fixingVal = domain.lb(var);
			else
				fixingVal = domain.ub(var);
		}
		else
		{
			FP_ASSERT(domain.type(var) == 'I');
			double xref_down = floorEps(xref[var]);
			double xref_up = ceilEps(xref[var]);
			double fracPart = fractionalPart(xref[var]);
			double value = (randVal >= fracPart) ? xref_down : xref_up;
			value = std::min(value, ub);
			value = std::max(value, lb);

			FP_ASSERT(isInteger(value, ABS_INT_TOL) && lb <= value && value <= ub);
			fixingVal = value;
		}

		return fixingVal;
	}

protected:
	const std::vector<double> &xref;
};

class BranchSimple : public DFSStrategy
{
public:
	BranchSimple(const MIPData &_data) : data(_data) {}
	void setup(const Domain &domain, RankerPtr ranker, ValuePtr _chooser)
	{
		chooser = _chooser;
		FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));
		sorted = (*ranker)(data, domain);
		FP_ASSERT(sorted.size() == (int)(data.mip.ncols - data.nContinuous));
		// compute inverse map
		invmap.resize(data.mip.ncols);
		std::fill(invmap.begin(), invmap.end(), -1);
		for (int idx = 0; idx < (int)sorted.size(); idx++)
			invmap[sorted[idx]] = idx;
	}

	std::vector<Branch> branch(const Domain &domain, bool nodeInfeas, const Branch &oldBranch) override
	{
		int startPos = (oldBranch.index >= 0) ? invmap[oldBranch.index] : 0;
		FP_ASSERT(startPos != -1);

		// Branch according to order
		for (int idx = startPos; idx < (int)sorted.size(); idx++)
		{
			int var = sorted[idx];
			FP_ASSERT(data.mip.xtype[var] != 'C');
			if (equal(domain.lb(var), domain.ub(var)))
				continue;

			/* Choose a preferred value */
			double lb = domain.lb(var);
			double ub = domain.ub(var);
			double value = (*chooser)(data, domain, var);
			if (equal(value, lb))
			{
				Branch preferred{var, 'U', domain.lb(var)};
				Branch other{var, 'L', domain.ub(var)};
				return {preferred, other};
			}
			else
			{
				Branch preferred{var, 'L', domain.ub(var)};
				Branch other{var, 'U', domain.lb(var)};
				return {preferred, other};
			}
		}
		return {};
	}

private:
	const MIPData &data;
	ValuePtr chooser;
	std::vector<int> sorted;
	std::vector<int> invmap;
};

class BranchNew : public DFSStrategy
{
public:
	BranchNew(const MIPData &_data) : data(_data) {}
	void setup(const Domain &domain, RankerPtr ranker, ValuePtr _chooser)
	{
		chooser = _chooser;
		FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));
		sorted = (*ranker)(data, domain);
		FP_ASSERT(sorted.size() == (int)(data.mip.ncols - data.nContinuous));
		// compute inverse map
		invmap.resize(data.mip.ncols);
		std::fill(invmap.begin(), invmap.end(), -1);
		for (int idx = 0; idx < (int)sorted.size(); idx++)
			invmap[sorted[idx]] = idx;
	}

	std::vector<Branch> branch(const Domain &domain, bool nodeInfeas, const Branch &oldBranch) override
	{
		int startPos = (oldBranch.index >= 0) ? invmap[oldBranch.index] : 0;
		FP_ASSERT(startPos != -1);

		// Branch according to order
		for (int idx = startPos; idx < (int)sorted.size(); idx++)
		{
			int var = sorted[idx];
			FP_ASSERT(data.mip.xtype[var] != 'C');

			/* Skip fixed variables. */
			if (equal(domain.lb(var), domain.ub(var)))
				continue;

			/* Choose a preferred value */
			const double lb = domain.lb(var);
			const double ub = domain.ub(var);
			const double value = (*chooser)(data, domain, var);

			FP_ASSERT(lb <= value && value <= ub);

			/* If value == lb we prefer x <= lb, so x == lb; if this is infeasible, we fix x == ub. */
			if (equal(value, lb))
			{
				/* var <= lb -> var == lb */
				Branch preferred{var, 'U', lb};
				/* var >= ub -> var == ub */
				Branch other{var, 'L', ub};
				return {preferred, other};
			}
			/* If value == ub we prefer x >= ub, so x == ub. */
			else if (equal(value, ub))
			{
				/* var >= ub -> var == ub */
				Branch preferred{var, 'L', ub};
				/* var <= lb -> var == lb */
				Branch other{var, 'U', lb};
				return {preferred, other};
				/* If the value lies in-between upper and lower we prefer fixing var = value. Then ony do we try var = upper and var = lower. */
			}
			else
			{
				std::vector<Branch> branches;
				FP_ASSERT(lb < value && value < ub);
				/* var = value */
				branches.emplace_back(var, 'B', value);

				if (!greaterThan(lb, value - 1.0))
				{
					/* var <= value - 1 and append var to end of sorted array */
					branches.emplace_back(var, 'U', value - 1.0);
				}

				if (!lessThan(ub, value + 1.0))
				{
					/* var >= value + 1 and append var to end of sorted array */
					branches.emplace_back(var, 'L', value + 1.0);
				}

				return branches;
			}
		}
		return {};
	}

private:
	const MIPData &data;
	ValuePtr chooser;
	std::vector<int> sorted;
	std::vector<int> invmap;
};

RankerPtr makeRanker(RankerType ranker, const Params &params, const MIPData &data)
{
	switch (ranker)
	{
	case RankerType::LR:
		return RankerPtr{new LR()};
	case RankerType::TYPE:
		return RankerPtr{new ByType()};
	case RankerType::TYPECL:
		return RankerPtr{new ByTypeCl()};
	case RankerType::LOCKS:
		return RankerPtr{new Locks()};
	case RankerType::CLIQUES:
		return RankerPtr{new ByCliques(params.seed)};
	case RankerType::CLIQUES2:
		return RankerPtr{new ByCliques2(params.seed)};
	case RankerType::RANDOM:
		return RankerPtr{new RandomOrder(params.seed)};
	case RankerType::REDCOSTS:
		return RankerPtr{new ByReducedCosts(params.seed)};
	case RankerType::DUALS:
		return RankerPtr{new ByDuals()};
	case RankerType::FRAC:
		return RankerPtr{new ByFrac()};
	case RankerType::DUALS_BREAK_FRAC:
		return RankerPtr{new DualsWithFracTieBreaker()};
	case RankerType::FRAC_BREAK_DUALS:
		return RankerPtr{new FracWithDualsTieBreaker()};
	case RankerType::FRAC_BREAK_REDCOSTS:
		return RankerPtr{new FracWithRedCostTieBreaker()};
	case RankerType::REDCOSTS_BREAK_FRAC:
		return RankerPtr{new RedCostWithFracTieBreaker(params.seed)};
	default:
	case RankerType::UNKNOWN:
		FP_ASSERT(false);
		consoleError("Ranker UNKNOWN");
		exit(1);
	}
}

ValuePtr makeValueChooser(ValueChooserType value_chooser, const Params &params, const MIPData &data)
{
	switch (value_chooser)
	{
	case ValueChooserType::GOOD_OBJ:
		return ValuePtr{new GoodObj()};
	case ValueChooserType::BAD_OBJ:
		return ValuePtr{new BadObj()};
	case ValueChooserType::RANDOM:
		return ValuePtr{new RandomValue(params.seed)};
	case ValueChooserType::LOOSE:
		return ValuePtr{new Loose()};
	case ValueChooserType::RANDOM_LP:
		return ValuePtr{new RandomRelaxation(params.seed, data.primals)};
	case ValueChooserType::UP:
		return ValuePtr{new AlwaysUp()};
	case ValueChooserType::DOWN:
		return ValuePtr{new AlwaysDown()};
	case ValueChooserType::RANDOM_UP_DOWN:
		return ValuePtr{new RandomUpDown(params.seed)};
	case ValueChooserType::ROUND_INT:
		return ValuePtr{new RoundInt(data.primals)};
	case ValueChooserType::SPLIT:
		return ValuePtr{new Split()};
	default:
	case ValueChooserType::UNKNOWN:
		FP_ASSERT(false);
		consoleError("ValueChooser UNKNOWN");
		exit(1);
	}
}
