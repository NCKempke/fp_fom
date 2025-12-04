/**
 * @file cpxmodel.h
 * @brief Implementation of MIPModelI for CPLEX
 *
 * @author Domenico Salvagnin <dominiqs at gmail dot com>
 * @contributor Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2019–2025
 *
 * Copyright 2019 Domenico Salvagnin
 * Copyright 2025 Nils-Christian Kempke
 */

#include "cpxmodel.h"

#ifdef HAS_COPT
#include "coptmodel.h"
#endif
#ifdef HAS_XPRESS
#include "xprsmodel.h"
#endif
#ifdef HAS_GUROBI
#include "gurobimodel.h"
#endif

#include <stdexcept>
#include <signal.h>
#include <cstring>
#include <iostream>

int CPXModel_UserBreak = 0;

static void userSignalBreak(int signum)
{
	CPXModel_UserBreak = 1;
}

static void throwCplexError(CPXCENVptr env, int status)
{
	const unsigned int BUF_SIZE = 4096;
	char errmsg[BUF_SIZE];
	CPXgeterrorstring(env, status, errmsg);
	int trailer = std::strlen(errmsg) - 1;
	if (trailer >= 0)
		errmsg[trailer] = '\0';
	throw std::runtime_error(errmsg);
}

/* Make a call to a Cplex API function checking its return status */
template <typename Func, typename... Args>
void CPX_CALL(Func cpxfunc, CPXENVptr env, Args &&...args)
{
	int status = cpxfunc(env, std::forward<Args>(args)...);
	if (status)
		throwCplexError(env, status);
}

CPXModel::CPXModel()
{
	int status = 0;

	env = CPXopenCPLEX(&status);
	if (status)
		throwCplexError(nullptr, status);

	lp = CPXcreateprob(env, &status, "");
	if (status)
	{
		CPXcloseCPLEX(&env);
		throwCplexError(nullptr, status);
	}
}

CPXModel::CPXModel(CPXENVptr _env, CPXLPptr _lp, bool _ownEnv, bool _ownLP) : env(_env), lp(_lp), ownEnv(_ownEnv), ownLP(_ownLP)
{
	FP_ASSERT(env && lp);
}

CPXModel::~CPXModel()
{
	if (restoreSignalHandler)
		handleCtrlC(false);
	if (ownLP)
		CPXfreeprob(env, &lp);
	if (ownEnv)
		CPXcloseCPLEX(&env);
}

/* Read/Write */
void CPXModel::readModel(const std::string &filename)
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXreadcopyprob, env, lp, filename.c_str(), nullptr);
}

void CPXModel::writeModel(const std::string &filename, const std::string &format) const
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXwriteprob, env, lp, filename.c_str(), nullptr);
}

void CPXModel::writeSol(const std::string &filename) const
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXsolwrite, env, lp, filename.c_str());
}

/* Solve */
void CPXModel::lpopt(char method, double tol, double gapTol)
{
	FP_ASSERT(env && lp);

	switch (method)
	{
	case 's':
	{
		/* Simplex. */
		CPX_CALL(CPXlpopt, env, lp);
		break;
	}
	case 'p':
	{
		/* Primal. */
		CPX_CALL(CPXprimopt, env, lp);
		break;
	}
	case 'd':
	{
		/* Dual simplex. */
		CPX_CALL(CPXdualopt, env, lp);
		break;
	}
	case 'b':
	{
		/* Barrier without crossover. */
		dblParam(DblParam::BarrierGap, gapTol);
		intParam(IntParam::Crossover, 0);
		CPX_CALL(CPXbaropt, env, lp);
		break;
	}
	case 'c':
	{
		/*Barrier with crossover. */
		dblParam(DblParam::BarrierGap, gapTol);
		intParam(IntParam::Crossover, 1);
		CPX_CALL(CPXbaropt, env, lp);
		break;
	}
	default:
		throw std::runtime_error("Unexpected method for lpopt");
	}

	/* Reset parameters to defaults. */
	intParam(IntParam::Crossover, 1);
	dblParam(DblParam::BarrierGap, 1e-8);
}

void CPXModel::mipopt()
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXmipopt, env, lp);
}

void CPXModel::presolve()
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXpresolve, env, lp, CPX_ALG_NONE);
}

void CPXModel::postsolve()
{
	FP_ASSERT(env && lp);
	// no-op in CPLEX
}

std::vector<double> CPXModel::postsolveSolution(const std::vector<double> &preX) const
{
	// uncrush solution
	int n = ncols();
	std::vector<double> origX(n, 0.0);
	CPX_CALL(CPXuncrushx, env, lp, &origX[0], &preX[0]);
	return origX;
}

/* Get solution */
double CPXModel::objval() const
{
	FP_ASSERT(env && lp);
	double ret;
	CPX_CALL(CPXgetobjval, env, lp, &ret);
	return ret;
}

void CPXModel::sol(double *x, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	CPX_CALL(CPXgetx, env, lp, x, first, last);
}

void CPXModel::dual_sol(double *y) const
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXgetpi, env, lp, y, 0, nrows() - 1);
}

void CPXModel::reduced_costs(double *redcosts) const
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXgetdj, env, lp, redcosts, 0, ncols() - 1);
}

bool CPXModel::isPrimalFeas() const
{
	FP_ASSERT(env && lp);
	int primalFeas = 0;
	CPX_CALL(CPXsolninfo, env, lp, nullptr, nullptr, &primalFeas, nullptr);
	return (primalFeas > 0);
}

/* Parameters */
void CPXModel::handleCtrlC(bool flag)
{
	if (flag)
	{
		CPXModel_UserBreak = 0;
		previousHandler = ::signal(SIGINT, userSignalBreak);
		restoreSignalHandler = true;
		CPX_CALL(CPXsetterminate, env, &CPXModel_UserBreak);
	}
	else
	{
		if (restoreSignalHandler)
		{
			::signal(SIGINT, previousHandler);
			restoreSignalHandler = false;
			CPX_CALL(CPXsetterminate, env, nullptr);
		}
	}
}

bool CPXModel::aborted() const
{
	return CPXModel_UserBreak;
}

void CPXModel::seed(int seed)
{
	FP_ASSERT(env);
	CPX_CALL(CPXsetintparam, env, CPX_PARAM_RANDOMSEED, seed);
}

void CPXModel::logging(bool log)
{
	FP_ASSERT(env);
	if (log)
		CPX_CALL(CPXsetintparam, env, CPX_PARAM_SCRIND, CPX_ON);
	else
		CPX_CALL(CPXsetintparam, env, CPX_PARAM_SCRIND, CPX_OFF);
}

int CPXModel::intParam(IntParam which) const
{
	FP_ASSERT(env);
	int value;

	switch (which)
	{
	case IntParam::Crossover:
		CPX_CALL(CPXgetintparam, env, CPX_PARAM_SOLUTIONTYPE, &value);
		break;
	case IntParam::Threads:
		CPX_CALL(CPXgetintparam, env, CPX_PARAM_THREADS, &value);
		break;
	case IntParam::SolutionLimit:
		CPX_CALL(CPXgetintparam, env, CPX_PARAM_INTSOLLIM, &value);
		break;
	case IntParam::NodeLimit:
		CPX_CALL(CPXgetintparam, env, CPX_PARAM_NODELIM, &value);
		break;
	case IntParam::IterLimit:
		CPX_CALL(CPXgetintparam, env, CPX_PARAM_ITLIM, &value);
		break;
	default:
		throw std::runtime_error("Unknown integer parameter");
	}

	return (int)value;
}

void CPXModel::intParam(IntParam which, int value)
{
	FP_ASSERT(env);

	switch (which)
	{
	case IntParam::Crossover:
		FP_ASSERT(value == 1 || value == 0);
		if (value == 0)
		{
			CPX_CALL(CPXsetintparam, env, CPX_PARAM_SOLUTIONTYPE, CPX_NONBASIC_SOLN);
		}
		else
		{
			/* This is the default. Theoretically one can force the crossover with CPX_BASIC_SOLN. */
			CPX_CALL(CPXsetintparam, env, CPX_PARAM_SOLUTIONTYPE, CPX_AUTO_SOLN);
		}
		break;
	case IntParam::Threads:
		CPX_CALL(CPXsetintparam, env, CPX_PARAM_THREADS, value);
		break;
	case IntParam::SolutionLimit:
		CPX_CALL(CPXsetintparam, env, CPX_PARAM_INTSOLLIM, value);
		break;
	case IntParam::NodeLimit:
		CPX_CALL(CPXsetintparam, env, CPX_PARAM_NODELIM, value);
		break;
	case IntParam::IterLimit:
		CPX_CALL(CPXsetintparam, env, CPX_PARAM_ITLIM, value);
		break;
	default:
		throw std::runtime_error("Unknown integer parameter");
	}
}

double CPXModel::dblParam(DblParam which) const
{
	FP_ASSERT(env);
	double value;

	switch (which)
	{
	case DblParam::BarrierGap:
		CPX_CALL(CPXgetdblparam, env, CPX_PARAM_BAREPCOMP, &value);
		break;
	case DblParam::TimeLimit:
		CPX_CALL(CPXgetdblparam, env, CPX_PARAM_TILIM, &value);
		break;
	case DblParam::FeasibilityTolerance:
		CPX_CALL(CPXgetdblparam, env, CPX_PARAM_EPRHS, &value);
		break;
	case DblParam::IntegralityTolerance:
		CPX_CALL(CPXgetdblparam, env, CPX_PARAM_EPINT, &value);
		break;
	default:
		throw std::runtime_error("Unknown double parameter");
	}

	return value;
}

void CPXModel::dblParam(DblParam which, double value)
{
	FP_ASSERT(env);
	switch (which)
	{
	case DblParam::BarrierGap:
		CPX_CALL(CPXsetdblparam, env, CPX_PARAM_BAREPCOMP, value);
		break;
	case DblParam::TimeLimit:
		CPX_CALL(CPXsetdblparam, env, CPX_PARAM_TILIM, value);
		break;
	case DblParam::FeasibilityTolerance:
		CPX_CALL(CPXsetdblparam, env, CPX_PARAM_EPRHS, value);
		break;
	case DblParam::IntegralityTolerance:
		CPX_CALL(CPXsetdblparam, env, CPX_PARAM_EPINT, value);
		break;
	default:
		throw std::runtime_error("Unknown double parameter");
	}
}

int CPXModel::intAttr(IntAttr which) const
{
	FP_ASSERT(env && lp);
	int value;

	switch (which)
	{
	case IntAttr::Nodes:
		value = CPXgetnodecnt(env, lp);
		break;
	case IntAttr::NodesLeft:
		value = CPXgetnodeleftcnt(env, lp);
		break;
	case IntAttr::BarrierIterations:
		value = CPXgetbaritcnt(env, lp);
		break;
	case IntAttr::SimplexIterations:
		value = CPXgetitcnt(env, lp);
		break;
	default:
		throw std::runtime_error("Unknown integer attribute");
	}

	return value;
}

double CPXModel::dblAttr(DblAttr which) const
{
	FP_ASSERT(env && lp);
	double value;

	switch (which)
	{
	case DblAttr::MIPDualBound:
		CPX_CALL(CPXgetbestobjval, env, lp, &value);
		break;
	case DblAttr::DualObjective:
		CPX_CALL(CPXgetdblquality, env, lp, &value, CPX_DUAL_OBJ);
		break;
	case DblAttr::MaxCompSlack:
		CPX_CALL(CPXgetdblquality, env, lp, &value, CPX_MAX_COMP_SLACK);
		break;
	case DblAttr::MaxDualInfeas:
		CPX_CALL(CPXgetdblquality, env, lp, &value, CPX_MAX_DUAL_INFEAS);
		break;
	case DblAttr::MaxPrimalInfeas:
		CPX_CALL(CPXgetdblquality, env, lp, &value, CPX_MAX_PRIMAL_INFEAS);
		break;
	default:
		throw std::runtime_error("Unknown double attribute");
	}

	return value;
}

void CPXModel::intParamInternal(int which, int value)
{
	FP_ASSERT(env);
	CPX_CALL(CPXsetintparam, env, which, value);
}

void CPXModel::dblParamInternal(int which, double value)
{
	FP_ASSERT(env);
	CPX_CALL(CPXsetdblparam, env, which, value);
}

// bkat
//  int CPXModel::which(char solverparam)
//  {
//  	FP_ASSERT(env);

// 	int which;

// 	if (solverparam == 'S')
// 	{
// 		which = CPXPARAM_SolutionType;
// 	}
// 	else if (solverparam == 'A')
// 	{
// 		which = CPXPARAM_Advance;
// 	}

// 	return which;
// }

// //bkat
// int CPXModel::value(char solverparam)
// {
// 	FP_ASSERT(env);

// 	int value;

// 	if (solverparam == 'N')
// 	{
// 		value = CPX_NONBASIC_SOLN;
// 	}
// 	else if (solverparam == 'B')
// 	{
// 		value = CPX_BASIC_SOLN;
// 	}
// 	else if (solverparam == 'A')
// 	{
// 		value = CPX_AUTO_SOLN;
// 	}

// 	return value;
// }

/* Access model data */
int CPXModel::nrows() const
{
	FP_ASSERT(env && lp);
	return CPXgetnumrows(env, lp);
}

int CPXModel::ncols() const
{
	FP_ASSERT(env && lp);
	return CPXgetnumcols(env, lp);
}

int CPXModel::nnz() const
{
	int nnz;
	int tmp = 0;
	CPXgetrows(env, lp, &tmp, nullptr, nullptr, nullptr, 0, &nnz, 0, nrows() - 1);
	FP_ASSERT(nnz <= 0);
	return -nnz;
}

double CPXModel::objOffset() const
{
	FP_ASSERT(env && lp);
	double objOffset = 0.0;
	CPX_CALL(CPXgetobjoffset, env, lp, &objOffset);
	return objOffset;
}

ObjSense CPXModel::objSense() const
{
	FP_ASSERT(env && lp);
	int cpxobjsen = CPXgetobjsen(env, lp);
	return (cpxobjsen > 0) ? ObjSense::MIN : ObjSense::MAX;
}

void CPXModel::lbs(double *lb, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXgetlb, env, lp, lb, first, last);
}

void CPXModel::ubs(double *ub, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXgetub, env, lp, ub, first, last);
}

void CPXModel::objcoefs(double *obj, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXgetobj, env, lp, obj, first, last);
}

void CPXModel::ctypes(char *ctype, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);
	// CPX_CALL(CPXgetctype, env, lp, ctype, first, last);
	CPXgetctype(env, lp, ctype, first, last);
}

void CPXModel::sense(char *sense, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < nrows()));
	if (last == -1)
		last = nrows() - 1;
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXgetsense, env, lp, sense, first, last);
}

void CPXModel::rhs(double *rhs, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < nrows()));
	if (last == -1)
		last = nrows() - 1;
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXgetrhs, env, lp, rhs, first, last);
}

void CPXModel::row(int ridx, SparseVector &row, char &sense, double &rhs, double &rngval) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((ridx >= 0) && (ridx < nrows()));
	// get row nz
	int tmp = 0;
	int size;
	CPXgetrows(env, lp, &tmp, &tmp, 0, 0, 0, &size, ridx, ridx);
	// get coef
	size = -size;
	if (size)
	{
		row.resize(size);
		CPX_CALL(CPXgetrows, env, lp, &tmp, &tmp, row.idx(), row.coef(), size, &tmp, ridx, ridx);
	}
	else
	{
		row.clear();
	}
	// here to correctly handle empty constraints
	// get rhs
	CPX_CALL(CPXgetrhs, env, lp, &rhs, ridx, ridx);
	// get sense
	CPX_CALL(CPXgetsense, env, lp, &sense, ridx, ridx);
	// get rngval
	CPX_CALL(CPXgetrngval, env, lp, &rngval, ridx, ridx);
	// CPLEX treats ranged rows considering the constraint satisfied
	// if the linear expression is in the range [rhs, rhs+rngval].
	// However, we interpret ranged rows differently, and the allowed
	// range is [rhs-rngval,rhs] (in both cases, rngval >= 0).
	// So we might have to update the rhs
	if (sense == 'R')
	{
		FP_ASSERT(rngval >= 0.0);
		rhs += rngval;
	}
}

void CPXModel::rows(SparseMatrix &matrix) const
{
	FP_ASSERT(env && lp);
	// get nnz
	int tmp = 0;
	int m = nrows();
	int size;
	matrix.k = m;
	matrix.U = ncols();
	matrix.beg.resize(m);
	CPXgetrows(env, lp, &tmp, matrix.beg.data(), nullptr, nullptr, 0, &size, 0, m - 1);
	size = -size;
	FP_ASSERT(size >= 0);
	matrix.nnz = size;
	// get coefs
	if (size)
	{
		matrix.ind.resize(size);
		matrix.val.resize(size);
		CPX_CALL(CPXgetrows, env, lp, &tmp,
				 matrix.beg.data(), matrix.ind.data(), matrix.val.data(),
				 size, &tmp, 0, m - 1);
		// fill up cnt
		matrix.cnt.resize(m);
		for (int i = 0; i < (m - 1); i++)
		{
			matrix.cnt[i] = matrix.beg[i + 1] - matrix.beg[i];
		}
		matrix.cnt[m - 1] = matrix.nnz - matrix.beg[m - 1];
	}
	else
	{
		matrix.cnt.clear();
		matrix.ind.clear();
		matrix.val.clear();
	}
}

void CPXModel::col(int cidx, SparseVector &col, char &type, double &lb, double &ub, double &obj) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((cidx >= 0) && (cidx < ncols()));
	// get col nz
	int tmp = 0;
	int size;
	CPXgetcols(env, lp, &tmp, &tmp, 0, 0, 0, &size, cidx, cidx);
	// get coefs
	size = -size;
	if (size)
	{
		col.resize(size);
		CPX_CALL(CPXgetcols, env, lp, &tmp, &tmp, col.idx(), col.coef(), size, &tmp, cidx, cidx);
	}
	else
	{
		col.clear();
	}
	// here to correctly handle empty vars
	// get bounds
	CPX_CALL(CPXgetlb, env, lp, &lb, cidx, cidx);
	CPX_CALL(CPXgetub, env, lp, &ub, cidx, cidx);
	// get obj
	CPX_CALL(CPXgetobj, env, lp, &obj, cidx, cidx);
	// get type
	int status = CPXgetctype(env, lp, &type, cidx, cidx);
	if (status)
		type = 'C'; // cannot call CPXgetctype on an LP
}

void CPXModel::cols(SparseMatrix &matrix) const
{
	FP_ASSERT(env && lp);
	// get nnz
	int tmp = 0;
	int n = ncols();
	int size;
	matrix.beg.resize(n);
	matrix.k = n;
	matrix.U = nrows();
	CPXgetcols(env, lp, &tmp, matrix.beg.data(), nullptr, nullptr, 0, &size, 0, n - 1);
	size = -size;
	FP_ASSERT(size >= 0);
	matrix.nnz = size;
	// get coefs
	if (size)
	{
		matrix.ind.resize(size);
		matrix.val.resize(size);
		CPX_CALL(CPXgetcols, env, lp, &tmp,
				 matrix.beg.data(), matrix.ind.data(), matrix.val.data(),
				 size, &tmp, 0, n - 1);
		// fill up cnt
		matrix.cnt.resize(n);
		for (int j = 0; j < (n - 1); j++)
		{
			matrix.cnt[j] = matrix.beg[j + 1] - matrix.beg[j];
		}
		matrix.cnt[n - 1] = matrix.nnz - matrix.beg[n - 1];
	}
	else
	{
		matrix.cnt.clear();
		matrix.ind.clear();
		matrix.val.clear();
	}
}

void CPXModel::colNames(std::vector<std::string> &names, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);
	names.clear();
	int count = last - first + 1;
	std::vector<char> buffer;
	std::vector<char *> cnames(count, nullptr);
	int surplus;
	CPXgetcolname(env, lp, &cnames[0], nullptr, 0, &surplus, first, last);
	if (surplus)
	{
		buffer.resize(-surplus);
		CPX_CALL(CPXgetcolname, env, lp, &cnames[0], &buffer[0], buffer.size(), &surplus, first, last);
		for (int i = 0; i < count; i++)
			names.push_back(std::string(cnames[i]));
	}
	else
	{
		// no names
		for (int i = 0; i < count; i++)
			names.push_back("");
	}
}

void CPXModel::rowNames(std::vector<std::string> &names, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < nrows()));
	if (last == -1)
		last = nrows() - 1;
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);
	names.clear();
	int count = last - first + 1;
	std::vector<char> buffer;
	std::vector<char *> rnames(count, 0);
	int surplus;
	CPXgetrowname(env, lp, &rnames[0], nullptr, 0, &surplus, first, last);
	if (surplus)
	{
		buffer.resize(-surplus);
		CPX_CALL(CPXgetrowname, env, lp, &rnames[0], &buffer[0], buffer.size(), &surplus, first, last);
		for (int i = 0; i < count; i++)
			names.push_back(std::string(rnames[i]));
	}
	else
	{
		// no names
		for (int i = 0; i < count; i++)
			names.push_back("");
	}
}

void CPXModel::rangeVal(double *rngval, int first, int last) const
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < nrows()));
	if (last == -1)
		last = nrows() - 1;
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXgetrngval, env, lp, rngval, first, last);
}

/* Data modifications */
void CPXModel::addEmptyCol(const std::string &name, char ctype, double lb, double ub, double obj)
{
	FP_ASSERT(env && lp);
	char *cname = (char *)(name.c_str());
	const char *ctypeptr = (ctype == 'C') ? nullptr : &ctype; //< do not risk turning the model into a MIP
	CPX_CALL(CPXnewcols, env, lp, 1, &obj, &lb, &ub, ctypeptr, &cname);
}

void CPXModel::addCol(const std::string &name, const int *idx, const double *val, int cnt, char ctype, double lb, double ub, double obj)
{
	FP_ASSERT(env && lp);
	int matbeg = 0;
	char *cname = (char *)(name.c_str());
	if (cnt > 0)
	{
		FP_ASSERT(idx && val);
		CPX_CALL(CPXaddcols, env, lp, 1, cnt, &obj, &matbeg, idx, val, &lb, &ub, &cname);
		if (ctype != 'C')
		{
			int last = ncols() - 1;
			CPX_CALL(CPXchgctype, env, lp, 1, &last, &ctype);
		}
	}
	else
		CPX_CALL(CPXnewcols, env, lp, 1, &obj, &lb, &ub, &ctype, &cname);
}

void CPXModel::addRow(const std::string &name, const int *idx, const double *val, int cnt, char sense, double rhs, double rngval)
{
	FP_ASSERT(env && lp);
	int matbeg = 0;
	char *rname = (char *)(name.c_str());
	if (sense == 'R')
	{
		FP_ASSERT(rngval >= 0.0);
		// for ranged rows, we assue [rhs-rngval,rhs] while CPLEX uses [rhs, rhs+rngval]
		rhs -= rngval;
	}
	CPX_CALL(CPXaddrows, env, lp, 0, 1, cnt, &rhs, &sense, &matbeg, idx, val, nullptr, &rname);
	if (sense == 'R')
	{
		FP_ASSERT(rngval >= 0.0);
		int ridx = nrows() - 1;
		FP_ASSERT(ridx >= 0);
		CPX_CALL(CPXchgrngval, env, lp, 1, &ridx, &rngval);
	}
}

void CPXModel::delRow(int ridx)
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXdelrows, env, lp, ridx, ridx);
}

void CPXModel::delCol(int cidx)
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXdelcols, env, lp, cidx, cidx);
}

void CPXModel::delRows(int first, int last)
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < nrows()));
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXdelrows, env, lp, first, last);
}

void CPXModel::delCols(int first, int last)
{
	FP_ASSERT(env && lp);
	FP_ASSERT((first >= 0) && (first < ncols()));
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);
	CPX_CALL(CPXdelcols, env, lp, first, last);
}

void CPXModel::objSense(ObjSense objsen)
{
	FP_ASSERT(env && lp);
	CPXchgobjsen(env, lp, static_cast<int>(objsen));
}

void CPXModel::objOffset(double val)
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXchgobjoffset, env, lp, val);
}

void CPXModel::lb(int cidx, double val)
{
	FP_ASSERT(env && lp);
	FP_ASSERT((cidx >= 0) && (cidx < ncols()));
	char lu = 'L';
	CPX_CALL(CPXchgbds, env, lp, 1, &cidx, &lu, &val);
}

void CPXModel::lbs(int cnt, const int *cols, const double *values)
{
	FP_ASSERT(env && lp);
	std::vector<char> lu(cnt, 'L');
	CPX_CALL(CPXchgbds, env, lp, cnt, cols, &lu[0], values);
}

void CPXModel::ub(int cidx, double val)
{
	FP_ASSERT(env && lp);
	FP_ASSERT((cidx >= 0) && (cidx < ncols()));
	char lu = 'U';
	CPX_CALL(CPXchgbds, env, lp, 1, &cidx, &lu, &val);
}

void CPXModel::ubs(int cnt, const int *cols, const double *values)
{
	FP_ASSERT(env && lp);
	std::vector<char> lu(cnt, 'U');
	CPX_CALL(CPXchgbds, env, lp, cnt, cols, &lu[0], values);
}

void CPXModel::fixCol(int cidx, double val)
{
	FP_ASSERT(env && lp);
	FP_ASSERT((cidx >= 0) && (cidx < ncols()));
	char lu = 'B';
	CPX_CALL(CPXchgbds, env, lp, 1, &cidx, &lu, &val);
}

void CPXModel::objcoef(int cidx, double val)
{
	FP_ASSERT(env && lp);
	FP_ASSERT((cidx >= 0) && (cidx < ncols()));
	CPX_CALL(CPXchgobj, env, lp, 1, &cidx, &val);
}

void CPXModel::objcoefs(int cnt, const int *cols, const double *values)
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXchgobj, env, lp, cnt, cols, values);
}

void CPXModel::ctype(int cidx, char val)
{
	FP_ASSERT(env && lp);
	FP_ASSERT((cidx >= 0) && (cidx < ncols()));
	FP_ASSERT((val == 'B') || (val == 'I') || (val == 'C'));
	CPX_CALL(CPXchgctype, env, lp, 1, &cidx, &val);
}

void CPXModel::ctypes(int cnt, const int *cols, const char *values)
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXchgctype, env, lp, cnt, cols, values);
}

void CPXModel::switchToLP()
{
	FP_ASSERT(env && lp);
	CPX_CALL(CPXchgprobtype, env, lp, CPXPROB_LP);
}

/* Private interface */
CPXModel *CPXModel::clone_impl() const
{
	/* We have to create a new environment as well: one of the main reasons
	 * for cloning is to allow for parallel processing of clones, which is
	 * not possible if the env is shared among clones.
	 */
	FP_ASSERT(env && lp);
	int status = 0;

	CPXENVptr newEnv = CPXopenCPLEX(&status);
	if (status)
		throwCplexError(nullptr, status);

	CPXLPptr cloned = CPXcloneprob(newEnv, lp, &status);
	if (status)
	{
		CPXcloseCPLEX(&newEnv);
		throwCplexError(nullptr, status);
	}
	return new CPXModel(newEnv, cloned, true, true);
}

CPXModel *CPXModel::presolvedmodel_impl()
{
	int preStat;
	CPX_CALL(CPXgetprestat, env, lp, &preStat, nullptr, nullptr, nullptr, nullptr);

	if (preStat == 2) // reduced to empty problem
	{
		throw std::runtime_error("Problem too easy (solved in presolve stage)");
	}
	else if (preStat == 0) // return clone of this
	{
		// nothing to do
		return this;
	}
	else // return clone of presolved problem
	{
		int status = 0;

		CPXENVptr newEnv = CPXopenCPLEX(&status);
		if (status)
			throwCplexError(env, status);

		CPXCLPptr presolved_lp = nullptr;
		CPX_CALL(CPXgetredlp, env, lp, &presolved_lp);

		CPXLPptr cloned_presolved_lp = CPXcloneprob(newEnv, presolved_lp, &status);
		if (status)
			throwCplexError(env, status);

		return new CPXModel(newEnv, cloned_presolved_lp, true, true);
	}
	FP_ASSERT(false);
	return nullptr;
}

#ifdef HAS_XPRESS
MIPModelPtr CPXModel::convertToXPRESS()
{
	/* get data of this; TODO duplicated in COPT method */
	const int n_col = this->ncols();
	const int m_row = this->nrows();

	std::vector<double> xlb(n_col);
	std::vector<double> xub(n_col);
	std::vector<double> xobj(n_col);
	std::vector<char> xtype(n_col);

	std::vector<double> row_rhs(m_row);
	std::vector<double> rngval(m_row);
	std::vector<char> row_sense(m_row);

	SparseMatrix matrix;

	std::vector<std::string> col_names;
	std::vector<std::string> row_names;

	this->lbs(xlb.data());
	this->ubs(xub.data());
	this->objcoefs(xobj.data());
	this->ctypes(xtype.data());
	this->sense(row_sense.data());
	this->rhs(row_rhs.data());
	this->rows(matrix);
	this->rangeVal(rngval.data(), 0, m_row - 1);
	this->rowNames(row_names);
	this->colNames(col_names);

	/* create the XPRESS problem */
	MIPModelPtr xpress_model = std::make_shared<XPRSModel>();

	/* Copy rows and columns (including objective vlaues). */
	for (int j_col = 0; j_col < n_col; ++j_col)
	{
		if ((xtype[j_col] != 'C') && (xtype[j_col] != 'I') && (xtype[j_col] != 'B'))
			throw std::runtime_error("Unsupported variable type for FP");

		xpress_model->addEmptyCol(col_names[j_col], xtype[j_col], xlb[j_col], xub[j_col], xobj[j_col]);
	}

	for (int i_row = 0; i_row < m_row; ++i_row)
	{
		const int nnz = matrix.cnt[i_row];
		const int row_start = matrix.beg[i_row];
		const int *row_idx = &matrix.ind[row_start];
		const double *row_val = &matrix.val[row_start];

		xpress_model->addRow(row_names[i_row], row_idx, row_val, nnz, row_sense[i_row], row_rhs[i_row], rngval[i_row]);
	}
	/* Copy the objective offset. */
	double objoff = this->objOffset();
	xpress_model->objOffset(objoff);

	return xpress_model;
}
#else
MIPModelPtr CPXModel::convertToXPRESS()
{
	assert(false && "XPRESS has not been linked");
	return nullptr;
};
#endif

#ifdef HAS_COPT
MIPModelPtr CPXModel::convertToCOPT()
{
	/* get data of this; TODO duplicated in XPRESS method */
	const int n_col = this->ncols();
	const int m_row = this->nrows();

	std::vector<double> xlb(n_col);
	std::vector<double> xub(n_col);
	std::vector<double> xobj(n_col);
	std::vector<char> xtype(n_col);

	std::vector<double> row_rhs(m_row);
	std::vector<double> rngval(m_row);
	std::vector<char> row_sense(m_row);

	SparseMatrix matrix;

	std::vector<std::string> col_names;
	std::vector<std::string> row_names;

	this->lbs(xlb.data());
	this->ubs(xub.data());
	this->objcoefs(xobj.data());
	this->ctypes(xtype.data());
	this->sense(row_sense.data());
	this->rhs(row_rhs.data());
	this->rows(matrix);
	this->rangeVal(rngval.data(), 0, m_row - 1);
	this->rowNames(row_names);
	this->colNames(col_names);

	/* create the COPT problem */
	MIPModelPtr copt_model = std::make_shared<COPTModel>();

	/* Copy rows and columns (including objective vlaues). */
	for (int j_col = 0; j_col < n_col; ++j_col)
	{
		// TODO: COPT dies when given 1e20 bounds instead of infinity .. Not sure why. A bug?
		if ((xtype[j_col] != 'C') && (xtype[j_col] != 'I') && (xtype[j_col] != 'B'))
			throw std::runtime_error("Unsupported variable type for FP");

		copt_model->addEmptyCol(col_names[j_col], xtype[j_col], xlb[j_col] < -1e15 ? -COPT_INFINITY : xlb[j_col], xub[j_col] > 1e15 ? COPT_INFINITY : xub[j_col], xobj[j_col]);
	}

	for (int i_row = 0; i_row < m_row; ++i_row)
	{
		const int nnz = matrix.cnt[i_row];
		const int row_start = matrix.beg[i_row];
		const int *row_idx = &matrix.ind[row_start];
		const double *row_val = &matrix.val[row_start];

		copt_model->addRow(row_names[i_row], row_idx, row_val, nnz, row_sense[i_row], row_rhs[i_row], rngval[i_row]);
	}

	/* Copy the objective offset. */
	const double obj_offset = this->objOffset();
	copt_model->objOffset(obj_offset);

	return copt_model;
}
#else
MIPModelPtr CPXModel::convertToCOPT()
{
	assert(false && "COPT has not been linked");
	return nullptr;
};
#endif

#ifdef HAS_GUROBI
MIPModelPtr CPXModel::convertToGUROBI()
{
	/* get data of this; TODO duplicated in XPRESS method */
	const int n_col = this->ncols();
	const int m_row = this->nrows();

	std::vector<double> xlb(n_col);
	std::vector<double> xub(n_col);
	std::vector<double> xobj(n_col);
	std::vector<char> xtype(n_col);

	std::vector<double> row_rhs(m_row);
	std::vector<double> rngval(m_row);
	std::vector<char> row_sense(m_row);

	SparseMatrix matrix;

	std::vector<std::string> col_names;
	std::vector<std::string> row_names;

	this->lbs(xlb.data());
	this->ubs(xub.data());
	this->objcoefs(xobj.data());
	this->ctypes(xtype.data());
	this->sense(row_sense.data());
	this->rhs(row_rhs.data());
	this->rows(matrix);
	this->rangeVal(rngval.data(), 0, m_row - 1);
	this->rowNames(row_names);
	this->colNames(col_names);

	/* create the COPT problem */
	MIPModelPtr gurobi_model = std::make_shared<GUROBIModel>();

	/* Copy rows and columns (including objective vlaues). */
	for (int j_col = 0; j_col < n_col; ++j_col)
	{
		if ((xtype[j_col] != 'C') && (xtype[j_col] != 'I') && (xtype[j_col] != 'B'))
			throw std::runtime_error("Unsupported variable type for FP");

		gurobi_model->addEmptyCol(col_names[j_col], xtype[j_col], xlb[j_col], xub[j_col], xobj[j_col]);
	}

	for (int i_row = 0; i_row < m_row; ++i_row)
	{
		const int nnz = matrix.cnt[i_row];
		const int row_start = matrix.beg[i_row];
		const int *row_idx = &matrix.ind[row_start];
		const double *row_val = &matrix.val[row_start];

		gurobi_model->addRow(row_names[i_row], row_idx, row_val, nnz, row_sense[i_row], row_rhs[i_row], rngval[i_row]);
	}
	/* Copy the objective offset. */
	double objoff = this->objOffset();
	gurobi_model->objOffset(objoff);

	return gurobi_model;
}
#else
MIPModelPtr CPXModel::convertToGUROBI()
{
	assert(false && "Gurobi has not been linked");
	return nullptr;
};
#endif

MIPModelPtr CPXModel::convertTo(const std::string &solver)
{
	if (solver == "CPLEX")
		return this->clone();
#ifdef HAS_COPT
	else if (solver == "COPT")
		return convertToCOPT();
#endif
#ifdef HAS_XPRESS
	else if (solver == "XPRESS")
		return convertToXPRESS();
#endif
#ifdef HAS_GUROBI
	else if (solver == "GUROBI")
		return convertToGUROBI();
#endif

	else
		assert(false && "Not implemented!");

	return nullptr;
}
