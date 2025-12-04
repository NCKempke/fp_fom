/**
 * @file coptmodel.h
 * @brief Implementation of MIPModelI for COPT
 *
 * @author Ying Wang <wang at zib dot de>
 * @contributor Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2022-2025
 *
 * Copyright 2022 Ying Wang
 * Copyright 2025 Nils-Christian Kempke
 */

#include "coptmodel.h"
#include <stdexcept>
#include <signal.h>
#include <iostream>
#include <cstring>

int COPTModel_UserBreak = 0;

static void userSignalBreak(int signum)
{
	COPTModel_UserBreak = 1;
}

static void throwCoptError(const int errcode)
{
	char errmsg[COPT_BUFFSIZE];
	COPT_GetRetcodeMsg(errcode, errmsg, COPT_BUFFSIZE);
	int trailer = std::min(std::strlen(errmsg), static_cast<size_t>(COPT_BUFFSIZE - 1));
	if (trailer >= 0)
		errmsg[trailer] = '\0';
	throw std::runtime_error(errmsg);
}

/* Make a call to a Copt API function and check its return status */
template <typename Func, typename... Args>
void COPT_API_CALL(Func coptfunc, copt_prob *prob, Args &&...args)
{
	int status = coptfunc(prob, std::forward<Args>(args)...);
	if (status)
		throwCoptError(status);
}

COPTModel::COPTModel()
{
	int errcode = 0;

	errcode = COPT_CreateEnv(&env);
	if (errcode)
		throwCoptError(errcode);

	errcode = COPT_CreateProb(env, &prob);
	if (errcode)
		throwCoptError(errcode);
}

COPTModel::COPTModel(copt_env *_env, copt_prob *_prob) : env(_env), prob(_prob)
{
	FP_ASSERT(env && prob);
}

COPTModel::~COPTModel()
{
	COPT_DeleteProb(&prob);
	COPT_DeleteEnv(&env);
}

/* Read/Write */
void COPTModel::readModel(const std::string &filename)
{
	FP_ASSERT(prob);

	if (filename.ends_with(".lp") || filename.ends_with(".lp.gz"))
	{
		COPT_API_CALL(COPT_ReadLp, prob, filename.c_str());
	}
	else if (filename.ends_with(".mps") || filename.ends_with(".mps.gz"))
	{
		COPT_API_CALL(COPT_ReadMps, prob, filename.c_str());
	}
	else if (filename.ends_with(".mst") || filename.ends_with(".mst.gz"))
	{
		COPT_API_CALL(COPT_ReadMst, prob, filename.c_str());
	}
	else
	{
		std::runtime_error("Unknown read format for Copt");
	}
}

void COPTModel::writeModel(const std::string &filename, const std::string &format) const
{
	FP_ASSERT(prob);
	if (format == "" || format == "Lp")
	{
		COPT_API_CALL(COPT_WriteLp, prob, filename.c_str());
	}
	else if (format == "Mps")
	{
		COPT_API_CALL(COPT_WriteMps, prob, filename.c_str());
	}
	else if (format == "Mst")
	{
		COPT_API_CALL(COPT_WriteMst, prob, filename.c_str());
	}
	else
	{
		std::runtime_error("Unknown output format for Copt");
	}
}

void COPTModel::writeSol(const std::string &filename) const
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_WriteSol, prob, filename.c_str());
}

void COPTModel::lpopt(char method, double tol, double gapTol)
{
	const double oldBarTol = dblParam(DblParam::BarrierGap);
	double oldPDLPTol;

	FP_ASSERT(prob);

	COPT_API_CALL(COPT_GetDblParam, prob, COPT_DBLPARAM_PDLPTOL, &oldPDLPTol);

	switch (method)
	{
	case 's':
	{
		/* Simplex (will be dual). */
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LPMETHOD, 1);
		break;
	}
	case 'p':
	{
		throw std::runtime_error("COPT has no primal simplex setting.");
	}
	case 'd':
	{
		/* Dual simplex */
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LPMETHOD, 1);
		break;
	}
	case 'b':
	{
		/* Barrier, no crossover. */
		intParam(IntParam::Crossover, 0);
		dblParam(DblParam::BarrierGap, gapTol);
		COPT_API_CALL(COPT_SetDblParam, prob, "BarPrimalTol", tol);
		COPT_API_CALL(COPT_SetDblParam, prob, "BarDualTol", tol);
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LPMETHOD, 2);

		break;
	}
	case 'c':
	{
		/* Barrier with crossover. */
		intParam(IntParam::Crossover, 1);
		dblParam(DblParam::BarrierGap, gapTol);
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LPMETHOD, 2);

		break;
	}
	case 'f':
	{
		/* First order method = PDLP, no crossover. */
		intParam(IntParam::Crossover, 0);
		COPT_API_CALL(COPT_SetDblParam, prob, COPT_DBLPARAM_PDLPTOL, tol);
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LPMETHOD, 6);
		break;
	}
	default:
		throw std::runtime_error("Unexpected method for COPT lpopt");
	}

	COPT_API_CALL(COPT_SolveLp, prob);

	/* Reset lpMethod, barrier, and tolerances. */
	intParam(IntParam::Crossover, 1);
	dblParam(DblParam::BarrierGap, oldBarTol);
	COPT_API_CALL(COPT_SetDblParam, prob, "BarPrimalTol", 1e-8);
	COPT_API_CALL(COPT_SetDblParam, prob, "BarDualTol", 1e-8);
	COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LPMETHOD, -1);
	COPT_API_CALL(COPT_SetDblParam, prob, COPT_DBLPARAM_PDLPTOL, oldPDLPTol);
}

void COPTModel::mipopt()
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_Solve, prob);
}

void COPTModel::presolve()
{
	FP_ASSERT(prob);
	/* not implemented yet */
}

void COPTModel::postsolve()
{
	FP_ASSERT(prob);
	/* not implemented yet */
}

std::vector<double> COPTModel::postsolveSolution(const std::vector<double> &preX) const
{
	FP_ASSERT(prob);

	int n = ncols();
	std::vector<double> origX(n, 0.0);
	/* not implemented yet */

	return origX;
}

double COPTModel::objval() const
{
	FP_ASSERT(prob);
	double lpobjval;
	/* Attention: assume copt interface only solving lp problem, therefore here returned lp objective value */
	COPT_API_CALL(COPT_GetDblAttr, prob, COPT_DBLATTR_LPOBJVAL, &lpobjval);
	return lpobjval;
}

void COPTModel::sol(double *x, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	COPT_API_CALL(COPT_GetLpSolution, prob, x, nullptr, nullptr, nullptr);
}

void COPTModel::dual_sol(double *y) const
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_GetLpSolution, prob, nullptr, nullptr, y, nullptr);
}

void COPTModel::reduced_costs(double *redcost) const
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_GetLpSolution, prob, nullptr, nullptr, nullptr, redcost);
}

bool COPTModel::isPrimalFeas() const
{
	FP_ASSERT(prob);
	int lpstat = COPT_LPSTATUS_UNSTARTED;
	COPT_API_CALL(COPT_GetIntAttr, prob, COPT_INTATTR_LPSTATUS, &lpstat);
	if (lpstat == COPT_LPSTATUS_OPTIMAL || lpstat == COPT_LPSTATUS_NUMERICAL)
		return true;
	else
		return false;
}

/* Parameters */
void COPTModel::handleCtrlC(bool flag)
{
	/* not tested yet */
	if (flag)
	{
		COPTModel_UserBreak = 0;
		previousHandler = ::signal(SIGINT, userSignalBreak);
		restoreSignalHandler = true;
		COPT_API_CALL(COPT_Interrupt, prob);
	}
	else
	{
		if (restoreSignalHandler)
		{
			::signal(SIGINT, previousHandler);
			restoreSignalHandler = false;
			COPT_API_CALL(COPT_Interrupt, prob);
		}
	}
}

bool COPTModel::aborted() const
{
	return COPTModel_UserBreak;
}

void COPTModel::seed(int seed)
{
	FP_ASSERT(prob);
	/* not implemented yet */
}

void COPTModel::logging(bool log)
{
	FP_ASSERT(prob);
	if (log)
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LOGGING, 1);
	else
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_LOGGING, 0);
}

int COPTModel::intParam(IntParam which) const
{
	FP_ASSERT(prob);
	int value;

	switch (which)
	{
	case IntParam::Crossover:
		COPT_API_CALL(COPT_GetIntParam, prob, COPT_INTPARAM_CROSSOVER, &value);
		break;
	case IntParam::Threads:
		COPT_API_CALL(COPT_GetIntParam, prob, COPT_INTPARAM_THREADS, &value);
		break;
	case IntParam::SolutionLimit:
		/* maybe not available */
		value = -1;
		break;
	case IntParam::NodeLimit:
		COPT_API_CALL(COPT_GetIntParam, prob, COPT_INTPARAM_NODELIMIT, &value);
		break;
	case IntParam::IterLimit:
		COPT_API_CALL(COPT_GetIntParam, prob, COPT_INTPARAM_BARITERLIMIT, &value);
		break;
	default:
		throw std::runtime_error("Unknown integer parameter");
	}

	return (int)value;
}

void COPTModel::intParam(IntParam which, int value)
{
	FP_ASSERT(prob);

	switch (which)
	{
	case IntParam::Crossover:
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_CROSSOVER, value);
		break;
	case IntParam::Threads:
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_THREADS, value);
		break;
	case IntParam::SolutionLimit:
		/* maybe not available */
		break;
	case IntParam::NodeLimit:
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_NODELIMIT, value);
		break;
	case IntParam::IterLimit:
		COPT_API_CALL(COPT_SetIntParam, prob, COPT_INTPARAM_BARITERLIMIT, value);
		break;
	default:
		throw std::runtime_error("Unknown integer parameter");
	}
}

double COPTModel::dblParam(DblParam which) const
{
	FP_ASSERT(prob);

	double value;

	switch (which)
	{
	case DblParam::BarrierGap:
		COPT_API_CALL(COPT_GetDblParam, prob, "BarGapTol", &value);
		break;
	case DblParam::TimeLimit:
		COPT_API_CALL(COPT_GetDblParam, prob, COPT_DBLPARAM_TIMELIMIT, &value);
		break;
	case DblParam::FeasibilityTolerance:
		COPT_API_CALL(COPT_GetDblParam, prob, COPT_DBLPARAM_FEASTOL, &value);
		break;
	case DblParam::IntegralityTolerance:
		COPT_API_CALL(COPT_GetDblParam, prob, COPT_DBLPARAM_INTTOL, &value);
		break;
	default:
		throw std::runtime_error("Unknown double parameter");
	}

	return value;
}

void COPTModel::dblParam(DblParam which, double value)
{
	FP_ASSERT(prob);

	switch (which)
	{
	case DblParam::BarrierGap:
		COPT_API_CALL(COPT_SetDblParam, prob, "BarGapTol", value);
		break;
	case DblParam::TimeLimit:
		COPT_API_CALL(COPT_SetDblParam, prob, COPT_DBLPARAM_TIMELIMIT, value);
		break;
	case DblParam::FeasibilityTolerance:
		COPT_API_CALL(COPT_SetDblParam, prob, COPT_DBLPARAM_FEASTOL, value);
		break;
	case DblParam::IntegralityTolerance:
		COPT_API_CALL(COPT_SetDblParam, prob, COPT_DBLPARAM_INTTOL, value);
		break;
	default:
		throw std::runtime_error("Unknown double parameter");
	}
}

int COPTModel::intAttr(IntAttr which) const
{
	FP_ASSERT(prob);
	int value;

	switch (which)
	{
	case IntAttr::Nodes:
		COPT_API_CALL(COPT_GetIntAttr, prob, COPT_INTATTR_NODECNT, &value);
		break;
	case IntAttr::NodesLeft:
		/*maybe not available */
		break;
	case IntAttr::BarrierIterations:
		COPT_API_CALL(COPT_GetIntAttr, prob, COPT_INTATTR_BARRIERITER, &value);
		break;
	case IntAttr::SimplexIterations:
		COPT_API_CALL(COPT_GetIntAttr, prob, COPT_INTATTR_SIMPLEXITER, &value);
		break;
	default:
		throw std::runtime_error("Unknown integer attribute");
	}

	return value;
}

double COPTModel::dblAttr(DblAttr which) const
{
	FP_ASSERT(prob);
	double value;

	switch (which)
	{
	case DblAttr::MIPDualBound:
		COPT_API_CALL(COPT_GetDblAttr, prob, COPT_DBLATTR_BESTOBJ, &value);
		break;
	default:
		throw std::runtime_error("Unknown double attribute");
	}

	return value;
}

void COPTModel::intParamInternal(int which, int value)
{
	FP_ASSERT(prob);
	/* not implementable yet*/
}

void COPTModel::dblParamInternal(int which, double value)
{
	FP_ASSERT(prob);
	/* not implementable yet*/
}

/* Access model data */
/* mark as necessary */
int COPTModel::nrows() const
{
	FP_ASSERT(prob);
	int nrows = 0;
	COPT_API_CALL(COPT_GetIntAttr, prob, COPT_INTATTR_ROWS, &nrows);
	return nrows;
}

/* mark as necessary */
int COPTModel::ncols() const
{
	FP_ASSERT(prob);
	int ncols = 0;
	COPT_API_CALL(COPT_GetIntAttr, prob, COPT_INTATTR_COLS, &ncols);
	return ncols;
}

int COPTModel::nnz() const
{
	FP_ASSERT(prob);
	int nnz;
	int m = nrows();
	int tmp = 0;
	COPT_API_CALL(COPT_GetRows, prob, m, nullptr, nullptr, nullptr, nullptr, nullptr, tmp, &nnz);
	FP_ASSERT(nnz >= 0);
	return nnz;
}

double COPTModel::objOffset() const
{
	FP_ASSERT(prob);
	double objOffset = 0.0;
	COPT_API_CALL(COPT_GetDblAttr, prob, COPT_DBLATTR_OBJCONST, &objOffset);
	return objOffset;
}

ObjSense COPTModel::objSense() const
{
	FP_ASSERT(prob);
	int cpxobjsen;
	COPT_API_CALL(COPT_GetIntAttr, prob, COPT_INTATTR_OBJSENSE, &cpxobjsen);
	return (cpxobjsen > 0) ? ObjSense::MIN : ObjSense::MAX;
}

void COPTModel::lbs(double *lb, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);

	int const count = last - first + 1;
	std::vector<int> cidx(count);
	for (int i = 0; i < count; i++)
		cidx[i] = first + i;

	/*Attention: The function is supposed to be called for getting lower bounds of all columns,
	where first and last should be passed in terms of all columns. */
	COPT_API_CALL(COPT_GetColInfo, prob, COPT_DBLINFO_LB, count, cidx.data(), lb);
}

void COPTModel::ubs(double *ub, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);

	int const count = last - first + 1;
	std::vector<int> cidx(count);
	for (int i = 0; i < count; i++)
		cidx[i] = first + i;

	/*Attention: The function is supposed to be called for getting lower bounds of all columns,
	where first and last should be passed in terms of all columns. */
	COPT_API_CALL(COPT_GetColInfo, prob, COPT_DBLINFO_UB, count, cidx.data(), ub);
}

void COPTModel::objcoefs(double *obj, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);

	int const count = last - first + 1;
	std::vector<int> cidx(count);
	for (int i = 0; i < count; i++)
		cidx[i] = first + i;

	/*Attention: The function is supposed to be called for getting lower bounds of all columns,
	where first and last should be passed in terms of all columns. */
	COPT_API_CALL(COPT_GetColInfo, prob, COPT_DBLINFO_OBJ, count, cidx.data(), obj);
}
/* mark as necessary */
void COPTModel::ctypes(char *ctype, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);

	int const count = last - first + 1;
	std::vector<int> cidx(count);
	for (int i = 0; i < count; i++)
		cidx[i] = first + i;

	/*Attention: The function is supposed to be called for getting lower bounds of all columns,
	where first and last should be passed in terms of all columns. */
	COPT_API_CALL(COPT_GetColType, prob, count, cidx.data(), ctype);
}

void COPTModel::sense(char *sense, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < nrows()));
	if (last == -1)
		last = nrows() - 1;
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);

	int const count = last - first + 1;

	int const numberrows = nrows();
	std::vector<double> lbs(count);
	std::vector<double> ubs(count);

	/*Attention: The function is supposed to be called for getting lower bounds of all rows,
	where first and last should be passed in terms of all rows. */
	COPT_API_CALL(COPT_GetRowInfo, prob, COPT_DBLINFO_LB, numberrows, nullptr, &lbs[0]);
	COPT_API_CALL(COPT_GetRowInfo, prob, COPT_DBLINFO_UB, numberrows, nullptr, &ubs[0]);

	for (int i = 0; i < count; i++)
	{
		int ridx = first + i;
		if (lbs[ridx] == -COPT_INFINITY && ubs[ridx] < COPT_INFINITY)
		{
			sense[i] = 'L';
		}
		else if (lbs[ridx] > -COPT_INFINITY && ubs[ridx] == COPT_INFINITY)
		{
			sense[i] = 'G';
		}
		else if (lbs[ridx] == ubs[ridx] && lbs[ridx] > -COPT_INFINITY && ubs[ridx] < COPT_INFINITY)
		{
			sense[i] = 'E';
		}
		else if (lbs[ridx] < ubs[ridx] && lbs[ridx] > -COPT_INFINITY && ubs[ridx] < COPT_INFINITY)
		{
			sense[i] = 'R';
		}
		else
		{
			throw std::runtime_error("Unknown sense of linear constrains");
		}
	}
}

void COPTModel::rhs(double *rhs, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < nrows()));
	if (last == -1)
		last = nrows() - 1;
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);

	int const count = last - first + 1;
	std::vector<double> lbs(count);
	std::vector<double> ubs(count);
	std::vector<char> sense(count);
	/*Attention: The function is supposed to be called for getting lower bounds of all rows,
	where first and last should be passed in terms of all rows. */
	COPT_API_CALL(COPT_GetRowInfo, prob, COPT_DBLINFO_LB, count, nullptr, &lbs[0]);
	COPT_API_CALL(COPT_GetRowInfo, prob, COPT_DBLINFO_UB, count, nullptr, &ubs[0]);
	this->sense(sense.data(), first, last);

	for (int i = 0; i < count; i++)
	{

		switch (sense[i])
		{
		case 'G':
			rhs[i] = lbs[i];
			break;
		case 'L':
			rhs[i] = ubs[i];
			break;
		case 'E':
			rhs[i] = ubs[i]; // ubs[i]==lbs[i]
			break;
		case 'R':
			rhs[i] = ubs[i];
			break;
		default:
			throw std::runtime_error("Unknown sense of linear constrains");
		}
	}
}

void COPTModel::row(int ridx, SparseVector &row, char &sense, double &rhs, double &rngval) const
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::rows(SparseMatrix &matrix) const
{
	FP_ASSERT(prob);
	int tmp = 0;
	int m = nrows();
	int size;
	matrix.k = m;
	matrix.U = ncols();
	matrix.beg.resize(m);
	/* see documentation copt, the function should be called twice, first time call for identifying size */
	COPT_API_CALL(COPT_GetRows, prob, m, nullptr, nullptr, nullptr, nullptr, nullptr, tmp, &size);
	FP_ASSERT(size >= 0); // if size non-positive, should be size = -size;
	matrix.nnz = size;
	if (size)
	{
		matrix.ind.resize(size);
		matrix.val.resize(size);
		matrix.cnt.resize(m);
		/* see documentation copt, the second time call for obtaining cols data */
		COPT_API_CALL(COPT_GetRows, prob, m, nullptr, matrix.beg.data(), matrix.cnt.data(), matrix.ind.data(), matrix.val.data(), size, nullptr);
	}
	else
	{
		matrix.cnt.clear();
		matrix.ind.clear();
		matrix.val.clear();
	}
}
void COPTModel::col(int cidx, SparseVector &col, char &type, double &lb, double &ub, double &obj) const
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::cols(SparseMatrix &matrix) const
{
	FP_ASSERT(prob);
	int tmp = 0;
	int n = ncols();
	int size;
	matrix.beg.resize(n);
	matrix.k = n;
	matrix.U = nrows();
	/* see documentation copt, the function should be called twice, first time call for identifying size */
	COPT_API_CALL(COPT_GetCols, prob, n, nullptr, nullptr, nullptr, nullptr, nullptr, tmp, &size);
	FP_ASSERT(size >= 0); // if size non-positive, should be size = -size;
	matrix.nnz = size;
	if (size)
	{
		matrix.ind.resize(size);
		matrix.val.resize(size);
		matrix.cnt.resize(n);
		/* see documentation copt, the second time call for obtaining cols data */
		COPT_API_CALL(COPT_GetCols, prob, n, nullptr, matrix.beg.data(), matrix.cnt.data(), matrix.ind.data(), matrix.val.data(), size, nullptr);
	}
	else
	{
		matrix.cnt.clear();
		matrix.ind.clear();
		matrix.val.clear();
	}
}
void COPTModel::colNames(std::vector<std::string> &names, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < ncols()));
	if (last == -1)
		last = ncols() - 1;
	FP_ASSERT((last >= 0) && (last < ncols()));
	FP_ASSERT(first <= last);
	names.clear();
	int count = last - first + 1;
	std::vector<char> buffer;
	int pReqSize;
	int buffsize = 0;
	for (int i = 0; i < count; i++)
	{
		COPT_API_CALL(COPT_GetColName, prob, first + i, nullptr, 0, &pReqSize);
		if (pReqSize)
		{
			buffer.resize(pReqSize);
			buffer.clear();
			buffsize = pReqSize;
			COPT_API_CALL(COPT_GetColName, prob, first + i, &buffer[0], buffsize, &pReqSize);
			buffsize = 0;
			pReqSize = 0;
		}
		names.push_back(std::string(&buffer[0]));
	}
}
void COPTModel::rowNames(std::vector<std::string> &names, int first, int last) const
{
	FP_ASSERT(prob);
	FP_ASSERT((first >= 0) && (first < nrows()));
	if (last == -1)
		last = nrows() - 1;
	FP_ASSERT((last >= 0) && (last < nrows()));
	FP_ASSERT(first <= last);
	names.clear();
	int count = last - first + 1;
	std::vector<char> buffer;
	int pReqSize;
	int buffsize = 0;
	for (int i = 0; i < count; i++)
	{
		COPT_API_CALL(COPT_GetRowName, prob, first + i, nullptr, 0, &pReqSize);
		if (pReqSize)
		{
			buffer.resize(pReqSize);
			buffer.clear();
			buffsize = pReqSize;
			COPT_API_CALL(COPT_GetRowName, prob, first + i, &buffer[0], buffsize, &pReqSize);
			buffsize = 0;
			pReqSize = 0;
		}
		names.push_back(std::string(&buffer[0]));
	}
}

void COPTModel::rangeVal(double *rngval, int first, int last) const
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}

/* Data modifications */
void COPTModel::addEmptyCol(const std::string &name, char ctype, double lb, double ub, double obj)
{
	FP_ASSERT(prob);
	char *cname = (char *)(name.c_str());
	/* copt addCol needs information about sparse structure of the column */
	/* since it is an empty column, assume zero number of non-zero item, empty idx list and emtpy value list */
	COPT_API_CALL(COPT_AddCol, prob, obj, 0, nullptr, nullptr, ctype, lb, ub, cname);
}

void COPTModel::addCol(const std::string &name, const int *idx, const double *val, int cnt, char ctype, double lb, double ub, double obj)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}

void COPTModel::addRow(const std::string &name, const int *idx, const double *val, int cnt, char sense, double rhs, double rngval)
{
	FP_ASSERT(prob);
	char *rname = (char *)(name.c_str());
	switch (sense)
	{
	case 'G':
		COPT_API_CALL(COPT_AddRow, prob, cnt, idx, val, 0, rhs, COPT_INFINITY, rname);
		break;
	case 'L':
		COPT_API_CALL(COPT_AddRow, prob, cnt, idx, val, 0, -COPT_INFINITY, rhs, rname);
		break;
	case 'E':
		COPT_API_CALL(COPT_AddRow, prob, cnt, idx, val, 0, rhs, rhs, rname);
		break;
	case 'R':
		COPT_API_CALL(COPT_AddRow, prob, cnt, idx, val, 0, rhs - rngval, rhs, rname);
		break;
	default:
		throw std::runtime_error("Unknown sense of linear constrains");
	}
}
void COPTModel::delRow(int ridx)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::delCol(int cidx)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::delRows(int first, int last)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::delCols(int first, int last)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::objSense(ObjSense objsen)
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_SetObjSense, prob, static_cast<int>(objsen));
}
void COPTModel::objOffset(double val)
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_SetObjConst, prob, val);
}
void COPTModel::lb(int cidx, double val)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::lbs(int cnt, const int *cols, const double *values)
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_SetColLower, prob, cnt, cols, values);
}
void COPTModel::ub(int cidx, double val)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::ubs(int cnt, const int *cols, const double *values)
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_SetColUpper, prob, cnt, cols, values);
}
void COPTModel::fixCol(int cidx, double val)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::objcoef(int cidx, double val)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}
void COPTModel::objcoefs(int cnt, const int *cols, const double *values)
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_SetColObj, prob, cnt, cols, values);
}

void COPTModel::ctype(int cidx, char val)
{
	FP_ASSERT(prob);
	/* not implemented yet*/
}

/* mark as necessary */
void COPTModel::ctypes(int cnt, const int *cols, const char *values)
{
	FP_ASSERT(prob);
	COPT_API_CALL(COPT_SetColType, prob, cnt, cols, values);
}

void COPTModel::switchToLP()
{
	FP_ASSERT(prob);
	int count = ncols();
	std::vector<char> ctypesval(count);
	for (int i = 0; i < count; i++)
		ctypesval[i] = 'C';
	this->ctypes(count, nullptr, ctypesval.data());
}

/* private  */
COPTModel *COPTModel::clone_impl() const
{
	FP_ASSERT(prob);
	int errcode = 0;

	copt_prob *cloned = nullptr;
	copt_env *clonedenv = nullptr;

	errcode = COPT_CreateEnv(&clonedenv);
	if (errcode)
		throwCoptError(errcode);

	errcode = COPT_CreateProb(clonedenv, &cloned);
	if (errcode)
		throwCoptError(errcode);

	errcode = COPT_CreateCopy(prob, &cloned);
	if (errcode)
		throwCoptError(errcode);

	std::unique_ptr<COPTModel> cloned_coptmodel(new COPTModel(clonedenv, cloned));
	return cloned_coptmodel.release();
}

COPTModel *COPTModel::presolvedmodel_impl()
{
	/* not implemented yet*/
	return nullptr;
}
