/**
 * @file   linear_propagator.h
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 */

#pragma once

#include <queue>
#include "propagation.h"
#include "mip.h"

/* Propagates generic linear contraints */
class LinearPropagator : public PropagatorI
{
public:
	LinearPropagator(const MIPInstance& mip);
	std::string name() const override { return "LinearPropagator"; }
	void init(const Domain &domain) override;
	void update(const Domain &domain, Domain::iterator mark) override;
	void undo(const Domain &domain, Domain::iterator mark) override;
	void propagate(PropagationEngine &engine, Domain::iterator mark, bool initialProp) override;
	void commit(const Domain &domain) override;

private:
	// Matrix data
	const MIPInstance& mip;
	// State (note: activities are maintained by the engine itself!)
	Domain::iterator lastUpdated;
	struct State
	{
		double diameter = 0.0;
		bool infeas = false;
	};
	std::vector<State> states;
	std::vector<int> firstNonBin;
	// Helpers
	State computeState(const Domain &domain, SparseMatrix::view_type row) const;
	void undoBoundChange(const Domain &domain, const BoundChange &bdchg);
	void propagateOneRowLessThan(PropagationEngine &engine, int i, double mult, double bound);
	void propagateOneRow(PropagationEngine &engine, int i);
};
