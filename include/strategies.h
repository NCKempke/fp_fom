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
	Ranker(uint64_t _seed = 0) : seed{_seed} {}
	virtual ~Ranker() {}
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) = 0;

protected:
	uint64_t seed;
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
		std::vector<int> rowsSortedByDual(data.mip.nRows);
		std::vector<int> colsSorted(data.mip.ncols - data.nContinuous);

		FP_ASSERT(data.duals.size() >= data.mip.nRows);

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
		const auto &rowCnt = matrix.cnt;
		const auto &rowInd = matrix.ind;

		FP_ASSERT(data.reduced_costs.size() >= data.mip.ncols);
		FP_ASSERT(rowBeg.size() == data.mip.nRows);

		for (int iRow = 0; iRow < data.mip.nRows; ++iRow)
		{
			int nBinInt = 0;
			/* Extract the unmarked binaries and integers. */
			for (int iNz = rowBeg[iRow]; iNz < rowBeg[iRow] + rowCnt[iRow]; ++iNz)
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

class ByIntegrality : public Ranker
{
public:
	ByIntegrality(const std::vector<double> &_xref) : xref(_xref) {}
	virtual std::vector<int> operator()(const MIPData &data, const Domain &domain) override
	{
		FP_ASSERT(data.mip.ncols == (data.nBinaries + data.nIntegers + data.nContinuous));
		FP_ASSERT(!xref.empty());
		std::vector<int> sorted;
		std::vector<double> score(data.mip.ncols);
		for (int j = 0; j < data.mip.ncols; j++)
		{
			if (data.mip.xtype[j] == 'C')
				continue;
			score[j] = integralityViolation(xref[j]);
			sorted.push_back(j);
		}
		std::sort(sorted.begin(), sorted.end(), [&](int v1, int v2)
				  { return (score[v1] < score[v2]); });
		return sorted;
	}

protected:
	std::vector<double> xref;
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
		int sentinel = -2 * data.mip.nRows;
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
			// if (data.uplocks[var] > data.dnlocks[var])  score[var] = -data.dnlocks[var];
			// else                                        score[var] = -data.uplocks[var];
		}

		// sort by increasing score
		std::vector<int> sorted(data.mip.ncols);
		std::iota(sorted.begin(), sorted.end(), 0);
		std::sort(sorted.begin(), sorted.end(), [&](int v1, int v2)
				  { return (score[v1] > score[v2]); });

		// drop continuous if any
		if (data.nContinuous)
		{
			int start = data.mip.ncols - data.nContinuous;
			FP_ASSERT(std::all_of(sorted.begin() + start, sorted.end(), [&](int v)
								  { return (score[v] == sentinel); }));
			sorted.erase(sorted.begin() + start, sorted.end());
		}

		// TODO: randomize within same score?
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
		// std::cout << var << " has bounds [" << domain.lb(var) << ", " << domain.ub(var) << "]\n";
		return std::floor((domain.ub(var) + domain.lb(var)) / 2.0);
	}
};

class RandomUpDown : public ValueChooser
{
public:
	RandomUpDown(uint64_t seed) : ValueChooser{seed} {};

	virtual double operator()(const MIPData &data, const Domain &domain, int var) override
	{
		std::srand(static_cast<unsigned int>(std::time(nullptr)));

		// Generate a random bool by taking the modulus of 2
		bool randomBool = std::rand() % 3;

		if (randomBool)
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

			FP_ASSERT(isInteger(value) && lb <= value && value <= ub);
			fixingVal = value;
		}

		return fixingVal;
	}

protected:
	const std::vector<double> &xref;
};

class StaticStrategy : public DFSStrategy
{
public:
	StaticStrategy(const MIPData &_data) : data(_data) {}
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
	case RankerType::LOCKS:
		return RankerPtr{new Locks()};
	case RankerType::RANDOM:
		return RankerPtr{new RandomOrder(params.seed)};
	case RankerType::REDCOSTS:
		return RankerPtr{new ByReducedCosts(params.seed)};
	case RankerType::TYPE:
		return RankerPtr{new ByType()};
	case RankerType::DUALS:
		return RankerPtr{new ByDuals()};
	case RankerType::FRAC:
		return RankerPtr{new ByFrac()};
	default:
		FP_ASSERT(false);
		return RankerPtr{};
	}
}

ValuePtr makeValueChooser(ValueChooserType value_chooser, const Params &params, const MIPData &data)
{
	switch (value_chooser)
	{
	case ValueChooserType::BAD_OBJ:
		return ValuePtr{new BadObj()};
	case ValueChooserType::DOWN:
		return ValuePtr{new AlwaysDown()};
	case ValueChooserType::GOOD_OBJ:
		return ValuePtr{new GoodObj()};
	case ValueChooserType::LOOSE:
		return ValuePtr{new Loose()};
	case ValueChooserType::RANDOM:
		return ValuePtr{new RandomValue(params.seed)};
	case ValueChooserType::RANDOM_LP:
		return ValuePtr{new RandomRelaxation(params.seed, data.primals)};
	case ValueChooserType::RANDOM_UP_DOWN:
		return ValuePtr{new RandomUpDown(params.seed)};
	case ValueChooserType::ROUND_INT:
		return ValuePtr{new RoundInt(data.primals)};
	case ValueChooserType::SPLIT:
		return ValuePtr{new Split()};
	case ValueChooserType::UP:
		return ValuePtr{new AlwaysUp()};
	default:
		FP_ASSERT(false);
		return ValuePtr{};
	}
}
