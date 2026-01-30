/**
 * @file   propagation.h
 * @brief  Constraint Propagation API
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 */

#pragma once

#include <vector>
#include <string>
#include <span>
#include <memory>
#include "mip.h"
#include "index_set.h"

/** Bound change data structure */

struct BoundChange
{
public:
	enum class Type : uint8_t
	{
		SHIFT = 0,
		UPPER,
		LOWER
	};
	BoundChange() = default;
	BoundChange(int j, Type t, double v, double old) : var(j), type(t), value(v), oldValue(old) {}
	Type type;		 /**< bound change type */
	int var;		 /**< column index */
	double value;	 /**< new value */
	double oldValue; /**< old value */
};

/** @brief domain data structure.
 *
 * This keeps a copy of the domain (i.e., bounds and type) of variables.
 * It is on this copy that propagation operates.
 */

class Domain
{
public:
	/* Initialize a domain from given bounds and variable types */
	void init(std::span<const double> _lb, std::span<const double> _ub, std::span<const char> _xtype);
	/* Simple getters */
	int ncols() const { return (int)xtype.size(); }
	inline double lb(int var) const { return xlb[var]; }
	inline double ub(int var) const { return xub[var]; }
	inline char type(int var) const { return xtype[var]; }
	inline std::span<const double> lbs() const { return xlb; }
	inline std::span<const double> ubs() const { return xub; }
	/* Checks if new candidate bounds would be accepted gvie current bounds and parameters */
	bool isNewLowerBoundAcceptable(int var, double newBound) const;
	bool isNewUpperBoundAcceptable(int var, double newBound) const;
	/* Change a bound on a variable */
	bool changeLowerBound(int var, double newBound);
	bool changeUpperBound(int var, double newBound);
	/* Shift a fixed variable to a new value */
	void shift(int var, double newValue);
	/* API for keeping track of changes since initialization */
	/* We cannot use a regular std::vector::iterator here, as that can be invalidated
	 * if memory is reallocated on insert. So we use the actual size instead.
	 * The price to pay is that we always need the domain reference as well,
	 * hence it is not an iterator in the STL sense...
	 */
	using iterator = size_t;
	inline iterator changesBegin() const { return 0; }
	inline iterator changesEnd() const { return stack.size(); }
	const BoundChange &change(iterator itr) const { return stack[itr]; }
	inline iterator mark() const { return changesEnd(); }
	void undo(iterator mark);
	const BoundChange &lastChange() const { return stack.back(); }
	void undoLast();
	/* Commit to current set of bounds changes */
	void commit() { stack.clear(); }
	/* Domain (numerical parameters) */
	double feasTol = 1e-5;
	double zeroTol = 1e-9;
	double infinity = 1e8;
	double minContRed = 0.1;
	/* Variable names for debugging */
	std::vector<std::string> cNames;

private:
	std::vector<double> xlb;
	std::vector<double> xub;
	std::vector<char> xtype;
	std::vector<BoundChange> stack;
};

/* Forward declaration of engine */
class PropagationEngine;

/* Propagator base class */
class PropagatorI
{
public:
	virtual ~PropagatorI() {}
	virtual std::string name() const = 0;
	virtual void init(const Domain &domain) {}
	virtual void update(const Domain &domain, Domain::iterator mark) {}
	virtual void undo(const Domain &domain, Domain::iterator mark) {}
	virtual void propagate(PropagationEngine &engine, Domain::iterator mark, bool initialProp) = 0;
	virtual void commit(const Domain &domain) {}
	bool enabled() const { return _enabled; }
	void enable() { _enabled = true; }
	void disable() { _enabled = false; }

public:
	Domain::iterator lastPropagated;
	bool _enabled = true;
};

using PropagatorPtr = std::shared_ptr<PropagatorI>;

/* Propagation Engine API */
class PropagationEngine
{
public:
	PropagationEngine(const MIPInstance& mip);
	/* Add a propagator to the engine */
	void add(PropagatorPtr prop) { propagators.push_back(prop); }
	/* Get a propagator by name */
	PropagatorPtr getPropagator(const std::string &name) const;
	/* Initialize domain and propagators */
	void init(std::span<const double> _lb, std::span<const double> _ub, std::span<const char> _xtype);
	/* Get a (const) reference to current domain */
	const Domain &getDomain() const { return domain; }
	/* Domain changes */
	bool changeLowerBound(int var, double newBound);
	bool changeUpperBound(int var, double newBound);
	bool fix(int var, double value);
	void shift(int var, double newValue);
	/* Actitivies */
	double getMinAct(int i) const { return minAct[i]; }
	double getMaxAct(int i) const { return maxAct[i]; }
	void recomputeRowActivity(int i);
	/* Violation */
	const IndexSet<int> &violatedRows() const { return violated; }
	double violation() const { return totViol; }
	/* Propagate until fixpoint or infeasibility */
	bool propagate(bool initialProp);
	/* Lightweight propagation: only direct implications of the current changes */
	bool directImplications();
	/* Support for backtracking */
	Domain::iterator mark() const { return domain.mark(); }
	void undo(Domain::iterator mark);
	/* Commit current set of changes */
	void commit();
	/* Enable/disable propagators */
	void enableAll();
	void disableAll();
	/* Engine parameters */
	int maxPasses = 100;
	void debugChecks() const;

protected:
	const MIPInstance& mip;

	Domain domain;
	std::vector<PropagatorPtr> propagators;
	// state
	std::vector<double> minAct;
	std::vector<double> maxAct;
	IndexSet<int> violated;
	double totViol = 0.0;
	// helpers
	void recomputeViolation();
	void computeActivity(int i, double &minAct, double &maxAct) const;

private:
	void debugCheckRow(int i) const;
};

/* Debugging aids */
void printChangesSinceMark(const Domain &domain, Domain::iterator mark);
