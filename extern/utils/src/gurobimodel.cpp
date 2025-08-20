/**
 * @file gurobimodel.cpp
 * @brief Implementation of MIPModelI for Gurobi
 *
 * @author Nils-Christian Kempke <nilskempke at gmail dot com>
 *
 * @date 2025
 *
 * Copyright 2025 Nils-Christian Kempke
 */

#include "gurobimodel.h"

#ifdef HAS_COPT
#include "coptmodel.h"
#endif
#ifdef HAS_XPRESS
#include "xprsmodel.h"
#endif
#ifdef HAS_CPLEX
#include "cpxmodel.h"
#endif

#include <stdexcept>
#include <signal.h>
#include <iostream>
#include <cstring>

int GUROBIModel_UserBreak = 0;

static void userSignalBreak(int signum)
{
    GUROBIModel_UserBreak = 1;
}

static void throwGurobiError(GRBenv *env)
{
    FP_ASSERT(env);
    char errmsg[1024];
    strcpy(errmsg, GRBgeterrormsg(env));
    int trailer = std::strlen(errmsg) - 1;
    if (trailer >= 0)
        errmsg[trailer] = '\0';
    throw std::runtime_error(errmsg);
}

/* Make a call to a Gurobi API function and check its return status */
template <typename Func, typename... Args>
void GUROBI_CALL(Func gurobifunc, GRBenv *env, GRBmodel *prob, Args &&...args)
{
    int status = gurobifunc(prob, std::forward<Args>(args)...);
    if (status)
        throwGurobiError(env);
}

/* Make a call to a Gurobi API function and check its return status */
template <typename Func, typename... Args>
void GUROBI_CALL_ENV(Func gurobifunc, GRBenv *env, Args &&...args)
{
    int status = gurobifunc(env, std::forward<Args>(args)...);
    if (status)
        throwGurobiError(env);
}

GUROBIModel::GUROBIModel()
{
    int errcode = 0;
    /* create an env */
    errcode = GRBloadenv(&env, NULL);
    if (errcode)
        throwGurobiError(env);

    /* create an empty model */
    errcode = GRBnewmodel(env, &prob, NULL, 0, NULL, NULL, NULL, NULL, NULL);

    if (errcode)
        throwGurobiError(env);
}

GUROBIModel::GUROBIModel(GRBenv *_env, GRBmodel *_prob) : env(_env), prob(_prob)
{
    FP_ASSERT(env && prob);
}

GUROBIModel::~GUROBIModel()
{
    FP_ASSERT(env && prob);

    if (presolvedProb)
        GRBfreemodel(presolvedProb);
    /* Free model */
    GRBfreemodel(prob);
    /* Free environment */
    GRBfreeenv(env);
}

/* Read/Write */
void GUROBIModel::readModel(const std::string &filename)
{
    FP_ASSERT(env && prob);
    GUROBI_CALL_ENV(GRBreadmodel, env, filename.c_str(), &prob);
}

void GUROBIModel::writeModel(const std::string &filename, const std::string &format) const
{
    FP_ASSERT(env && prob);
}

void GUROBIModel::writeSol(const std::string &filename) const
{
    FP_ASSERT(env && prob);
}

void GUROBIModel::lpopt(char method, double tol)
{
    FP_ASSERT(env && prob);

    switch (method)
    {
    case 's':
    {
        /* Run primal and dual simplex concurrently. */
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_METHOD, 3);
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_CONCURRENTMETHOD, 3);

        break;
    }
    case 'p':
    {
        /* Primal Simplex. */
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_METHOD, 0);

        break;
    }
    case 'd':
    {
        /* Dual simplex. */
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_METHOD, 1);

        break;
    }
    case 'b':
    {
        /* Barrier method without crossover. */
        intParam(IntParam::Crossover, 0);
        dblParam(DblParam::BarrierGap, tol);
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_METHOD, 2);

        break;
    }
    case 'c':
    {
        /* Barrier with crossover. */
        intParam(IntParam::Crossover, 1);
        dblParam(DblParam::BarrierGap, tol);
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_METHOD, 2);

        break;
    }
    default:
        throw std::runtime_error("Unexpected method for GUROBI lpopt");
    }

    GUROBI_CALL(GRBoptimize, GRBgetenv(prob), prob);

    /* Reset parameters to defaults. */
    dblParam(DblParam::BarrierGap, 1e-8);
    intParam(IntParam::Crossover, -1);
    GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_METHOD, -1);
    GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_CONCURRENTMETHOD, -1);
}

void GUROBIModel::mipopt()
{
    FP_ASSERT(env && prob);
    GUROBI_CALL(GRBoptimize, GRBgetenv(prob), prob);
}

void GUROBIModel::presolve()
{
    FP_ASSERT(env && prob);

    GUROBI_CALL(GRBpresolvemodel, env, prob, &presolvedProb);
}

void GUROBIModel::postsolve()
{
    FP_ASSERT(env && prob);
    /* not implemented yet */
}

std::vector<double> GUROBIModel::postsolveSolution(const std::vector<double> &preX) const
{
    FP_ASSERT(env && prob);

    int n = ncols();
    std::vector<double> origX(n, 0.0);
    /* not implemented yet */

    return origX;
}

double GUROBIModel::objval() const
{
    FP_ASSERT(env && prob);
    double lpobjval;

    return lpobjval;
}

void GUROBIModel::sol(double *x, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < ncols()));
    if (last == -1)
        last = ncols() - 1;
    FP_ASSERT((last >= 0) && (last < ncols()));
    int const count = last - first + 1;
    GUROBI_CALL(GRBgetdblattrarray, GRBgetenv(prob), prob, GRB_DBL_ATTR_X, first, count, x);
}

void GUROBIModel::dual_sol(double *y) const
{
    FP_ASSERT(env && prob);
    GUROBI_CALL(GRBgetdblattrarray, GRBgetenv(prob), prob, GRB_DBL_ATTR_PI, 0, nrows(), y);
}

void GUROBIModel::reduced_costs(double *reduced_costs) const
{
    FP_ASSERT(env && prob);
    GUROBI_CALL(GRBgetdblattrarray, GRBgetenv(prob), prob, GRB_DBL_ATTR_RC, 0, ncols(), reduced_costs);
}

bool GUROBIModel::isPrimalFeas() const
{
    FP_ASSERT(env && prob);
    int lpstat;
    GUROBI_CALL(GRBgetintattr, GRBgetenv(prob), prob, GRB_INT_ATTR_STATUS, &lpstat);
    if (lpstat == GRB_OPTIMAL)
        return true;
    else
        return false;
}

/* Parameters */
void GUROBIModel::handleCtrlC(bool flag)
{
    FP_ASSERT(env && prob);
}

bool GUROBIModel::aborted() const
{
    return GUROBIModel_UserBreak;
}

void GUROBIModel::seed(int seed)
{
    FP_ASSERT(env && prob);
    FP_ASSERT(seed >= 0);
    GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_SEED, seed);
}

void GUROBIModel::logging(bool log)
{
    FP_ASSERT(env && prob);
    if (log)
    {
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_LOGTOCONSOLE, 1);
    }
    else
    {
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_LOGTOCONSOLE, 0);
    }
}

int GUROBIModel::intParam(IntParam which) const
{
    FP_ASSERT(env && prob);
    double val;
    int value;

    switch (which)
    {
    case IntParam::Crossover:
        GUROBI_CALL_ENV(GRBgetintparam, GRBgetenv(prob), GRB_INT_PAR_CROSSOVER, &value);
        break;
    case IntParam::Threads:
        GUROBI_CALL_ENV(GRBgetintparam, GRBgetenv(prob), GRB_INT_PAR_THREADS, &value);
        break;
    case IntParam::SolutionLimit:
        GUROBI_CALL_ENV(GRBgetintparam, GRBgetenv(prob), GRB_INT_PAR_SOLUTIONLIMIT, &value);
        break;
    case IntParam::NodeLimit:
        GUROBI_CALL_ENV(GRBgetdblparam, GRBgetenv(prob), GRB_DBL_PAR_NODELIMIT, &val);
        value = static_cast<int>(val);
        break;
    case IntParam::IterLimit:
        GUROBI_CALL_ENV(GRBgetdblparam, GRBgetenv(prob), GRB_DBL_PAR_ITERATIONLIMIT, &val);
        value = static_cast<int>(val);
        break;
    default:
        throw std::runtime_error("Unknown integer parameter");
    }

    return value;
}

void GUROBIModel::intParam(IntParam which, int value)
{
    FP_ASSERT(env && prob);

    switch (which)
    {
    case IntParam::Crossover:
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_CROSSOVER, value);
        break;
    case IntParam::Threads:
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_THREADS, value);
        break;
    case IntParam::SolutionLimit:
        GUROBI_CALL_ENV(GRBsetintparam, GRBgetenv(prob), GRB_INT_PAR_SOLUTIONLIMIT, value);
        break;
    case IntParam::NodeLimit:
        GUROBI_CALL_ENV(GRBsetdblparam, GRBgetenv(prob), GRB_DBL_PAR_NODELIMIT, (double)value);
        break;
    case IntParam::IterLimit:
        GUROBI_CALL_ENV(GRBsetdblparam, GRBgetenv(prob), GRB_DBL_PAR_ITERATIONLIMIT, (double)value);
        break;
    default:
        throw std::runtime_error("Unknown integer parameter");
    }
}

double GUROBIModel::dblParam(DblParam which) const
{
    FP_ASSERT(env && prob);
    double value{};

    switch (which)
    {
    case DblParam::TimeLimit:
        GUROBI_CALL_ENV(GRBgetdblparam, GRBgetenv(prob), GRB_DBL_PAR_TIMELIMIT, &value);
        break;
    case DblParam::BarrierGap:
        GUROBI_CALL_ENV(GRBgetdblparam, GRBgetenv(prob), GRB_DBL_PAR_BARCONVTOL, &value);
        break;
    case DblParam::FeasibilityTolerance:
        GUROBI_CALL_ENV(GRBgetdblparam, GRBgetenv(prob), GRB_DBL_PAR_FEASIBILITYTOL, &value);
        break;
    case DblParam::IntegralityTolerance:
        GUROBI_CALL_ENV(GRBgetdblparam, GRBgetenv(prob), GRB_DBL_PAR_INTFEASTOL, &value);
        break;
    default:
        throw std::runtime_error("Unknown double parameter");
    }

    return value;
}

void GUROBIModel::dblParam(DblParam which, double value)
{
    switch (which)
    {
    case DblParam::TimeLimit:
        GUROBI_CALL_ENV(GRBsetdblparam, GRBgetenv(prob), GRB_DBL_PAR_TIMELIMIT, value);
        break;
    case DblParam::BarrierGap:
        GUROBI_CALL_ENV(GRBsetdblparam, GRBgetenv(prob), GRB_DBL_PAR_BARCONVTOL, value);
        break;
    case DblParam::FeasibilityTolerance:
        GUROBI_CALL_ENV(GRBsetdblparam, GRBgetenv(prob), GRB_DBL_PAR_FEASIBILITYTOL, value);
        break;
    case DblParam::IntegralityTolerance:
        GUROBI_CALL_ENV(GRBsetdblparam, GRBgetenv(prob), GRB_DBL_PAR_INTFEASTOL, value);
        break;
    default:
        throw std::runtime_error("Unknown double parameter");
    }
}

int GUROBIModel::intAttr(IntAttr which) const
{
    FP_ASSERT(env && prob);
    int value;
    FP_ASSERT(false);

    return value;
}

double GUROBIModel::dblAttr(DblAttr which) const
{
    FP_ASSERT(env && prob);
    double value;
    FP_ASSERT(false);

    return value;
}

void GUROBIModel::intParamInternal(int which, int value)
{
    FP_ASSERT(env && prob);
    FP_ASSERT(false);

    /* not implementable yet*/
}

void GUROBIModel::dblParamInternal(int which, double value)
{
    FP_ASSERT(env && prob);
    FP_ASSERT(false);

    /* not implementable yet*/
}

/* Access model data */
/* mark as necessary */
int GUROBIModel::nrows() const
{
    FP_ASSERT(env && prob);
    int nrows = 0;
    GUROBI_CALL(GRBgetintattr, GRBgetenv(prob), prob, "NumConstrs", &nrows);
    return nrows;
}

/* mark as necessary */
int GUROBIModel::ncols() const
{
    FP_ASSERT(env && prob);
    int ncols = 0;
    GUROBI_CALL(GRBgetintattr, GRBgetenv(prob), prob, "NumVars", &ncols);
    return ncols;
}

int GUROBIModel::nnz() const
{
    int nnz = -1;
    GUROBI_CALL(GRBgetintattr, GRBgetenv(prob), prob, "NumNZs", &nnz);
    FP_ASSERT(nnz >= 0);
    return nnz;
}

double GUROBIModel::objOffset() const
{
    FP_ASSERT(env && prob);
    double objOffset = 0.0;
    GUROBI_CALL(GRBgetdblattr, GRBgetenv(prob), prob, "ObjCon", &objOffset);
    return objOffset;
}

ObjSense GUROBIModel::objSense() const
{
    FP_ASSERT(env && prob);
    int objsen;
    GUROBI_CALL(GRBgetintattr, GRBgetenv(prob), prob, "ModelSense", &objsen);
    return (objsen > 0) ? ObjSense::MIN : ObjSense::MAX;
}

void GUROBIModel::lbs(double *lb, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < ncols()));
    if (last == -1)
        last = ncols() - 1;
    FP_ASSERT((last >= 0) && (last < ncols()));
    FP_ASSERT(first <= last);
    const int count = last - first + 1;
    GUROBI_CALL(GRBgetdblattrarray, GRBgetenv(prob), prob, "LB", first, count, lb);
}

void GUROBIModel::ubs(double *ub, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < ncols()));
    if (last == -1)
        last = ncols() - 1;
    FP_ASSERT((last >= 0) && (last < ncols()));
    FP_ASSERT(first <= last);
    const int count = last - first + 1;
    GUROBI_CALL(GRBgetdblattrarray, GRBgetenv(prob), prob, "UB", first, count, ub);
}

void GUROBIModel::objcoefs(double *obj, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < ncols()));
    if (last == -1)
        last = ncols() - 1;
    FP_ASSERT((last >= 0) && (last < ncols()));
    FP_ASSERT(first <= last);
    const int count = last - first + 1;
    GUROBI_CALL(GRBgetdblattrarray, GRBgetenv(prob), prob, "Obj", first, count, obj);
}

void GUROBIModel::ctypes(char *ctype, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < ncols()));
    if (last == -1)
        last = ncols() - 1;
    FP_ASSERT((last >= 0) && (last < ncols()));
    FP_ASSERT(first <= last);
    int const count = last - first + 1;
    GUROBI_CALL(GRBgetcharattrarray, GRBgetenv(prob), prob, "VType", first, count, ctype);
}

void GUROBIModel::sense(char *sense, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < nrows()));
    if (last == -1)
        last = nrows() - 1;
    FP_ASSERT((last >= 0) && (last < nrows()));
    FP_ASSERT(first <= last);
    const int count = last - first + 1;
    GUROBI_CALL(GRBgetcharattrarray, GRBgetenv(prob), prob, "Sense", first, count, sense);

    for (int i = 0; i < count; i++)
    {
        if (sense[i] == '<')
        {
            sense[i] = 'L';
        }
        else if (sense[i] == '>')
        {
            sense[i] = 'G';
        }
        else if (sense[i] == '=')
        {
            sense[i] = 'E';
        }
    }
}

void GUROBIModel::rhs(double *rhs, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < nrows()));
    if (last == -1)
        last = nrows() - 1;
    FP_ASSERT((last >= 0) && (last < nrows()));
    FP_ASSERT(first <= last);

    const int count = last - first + 1;
    GUROBI_CALL(GRBgetdblattrarray, GRBgetenv(prob), prob, "RHS", first, count, rhs);
}

void GUROBIModel::row(int ridx, SparseVector &row, char &sense, double &rhs, double &rngval) const
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
void GUROBIModel::rows(SparseMatrix &matrix) const
{
    FP_ASSERT(env && prob);

    int nconst = nrows();
    int size;
    matrix.beg.resize(nconst);
    matrix.k = nconst;
    matrix.U = ncols();
    /* see documentation gurobi, the function should be called twice, first time call for identifying nonzero element size */
    GUROBI_CALL(GRBgetconstrs, GRBgetenv(prob), prob, &size, nullptr, nullptr, nullptr, 0, nconst);
    FP_ASSERT(size >= 0); // if size non-positive, should be size = -size;
    matrix.nnz = size;
    if (size)
    {
        matrix.ind.resize(size);
        matrix.val.resize(size);
        matrix.cnt.resize(nconst);
        /* see documentation copt, the second time call for obtaining cols data */
        GUROBI_CALL(GRBgetconstrs, GRBgetenv(prob), prob, &size, matrix.beg.data(), matrix.ind.data(), matrix.val.data(), 0, nconst);
        // fill up cnt
        matrix.cnt.resize(nconst);
        for (int i = 0; i < (nconst - 1); i++)
        {
            matrix.cnt[i] = matrix.beg[i + 1] - matrix.beg[i];
        }
        matrix.cnt[nconst - 1] = matrix.nnz - matrix.beg[nconst - 1];
    }
    else
    {
        matrix.cnt.clear();
        matrix.ind.clear();
        matrix.val.clear();
    }
}
void GUROBIModel::col(int cidx, SparseVector &col, char &type, double &lb, double &ub, double &obj) const
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
void GUROBIModel::cols(SparseMatrix &matrix) const
{
    FP_ASSERT(env && prob);
    int nvars = ncols();
    int size;
    matrix.beg.resize(nvars);
    matrix.k = nvars;
    matrix.U = nrows();
    /* see documentation gurobi, the function should be called twice, first time call for identifying nonzero element size */
    GUROBI_CALL(GRBgetvars, GRBgetenv(prob), prob, &size, nullptr, nullptr, nullptr, 0, nvars);
    FP_ASSERT(size >= 0); // if size non-positive, should be size = -size;
    matrix.nnz = size;
    if (size)
    {
        matrix.ind.resize(size);
        matrix.val.resize(size);
        matrix.cnt.resize(nvars);
        /* see documentation copt, the second time call for obtaining cols data */
        GUROBI_CALL(GRBgetvars, GRBgetenv(prob), prob, &size, matrix.beg.data(), matrix.ind.data(), matrix.val.data(), 0, nvars);
        // fill up cnt
        for (int j = 0; j < (nvars - 1); j++)
        {
            matrix.cnt[j] = matrix.beg[j + 1] - matrix.beg[j];
        }
        matrix.cnt[nvars - 1] = matrix.nnz - matrix.beg[nvars - 1];
    }
    else
    {
        matrix.cnt.clear();
        matrix.ind.clear();
        matrix.val.clear();
    }
}

void GUROBIModel::colNames(std::vector<std::string> &names, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < ncols()));
    if (last == -1)
        last = ncols() - 1;
    FP_ASSERT((last >= 0) && (last < ncols()));
    FP_ASSERT(first <= last);

    names.clear();
    int const count = last - first + 1;
    std::vector<char> buffer;
    int pReqSize;
    int buffsize = 0;
    char *nameptr;

    for (int i = 0; i < count; i++)
    {
        GUROBI_CALL(GRBgetstrattrelement, GRBgetenv(prob), prob, "VarName", first + i, &nameptr);
        pReqSize = strlen(nameptr);

        if (pReqSize)
        {
            buffer.clear();
            buffsize = pReqSize;
            buffer.reserve(pReqSize);
            /*copy content of nameptr */
            while (*(nameptr) != '\0')
            {
                buffer.push_back(*(nameptr++));
            }
            buffer.push_back('\0');
        }
        names.push_back(std::string(&buffer[0]));
    }
}
void GUROBIModel::rowNames(std::vector<std::string> &names, int first, int last) const
{
    FP_ASSERT(env && prob);
    FP_ASSERT((first >= 0) && (first < nrows()));
    if (last == -1)
        last = nrows() - 1;
    FP_ASSERT((last >= 0) && (last < nrows()));
    FP_ASSERT(first <= last);

    names.clear();
    int const count = last - first + 1;
    std::vector<char> buffer;
    int pReqSize;
    int buffsize = 0;
    char *nameptr;

    for (int i = 0; i < count; i++)
    {
        GUROBI_CALL(GRBgetstrattrelement, GRBgetenv(prob), prob, "ConstrName", first + i, &nameptr);
        pReqSize = strlen(nameptr);

        if (pReqSize)
        {
            buffer.clear();
            buffsize = pReqSize;
            buffer.reserve(pReqSize);
            /*copy content of nameptr */
            while (*(nameptr) != '\0')
            {
                buffer.push_back(*(nameptr++));
            }
            buffer.push_back('\0');
        }
        names.push_back(std::string(&buffer[0]));
    }
}

void GUROBIModel::rangeVal(double *rngval, int first, int last) const
{
    FP_ASSERT(env && prob);
    /* Gurobi has no ranged constraints. */
    std::fill(rngval + first, rngval + last, 0.0);
}

/* Data modifications */
void GUROBIModel::addEmptyCol(const std::string &name, char ctype, double lb, double ub, double obj)
{
    FP_ASSERT(env && prob);
    char *cname = (char *)(name.c_str());
    GUROBI_CALL(GRBaddvar, GRBgetenv(prob), prob, 0, nullptr, nullptr, obj, lb, ub, ctype, cname);
}

void GUROBIModel::addCol(const std::string &name, const int *idx, const double *val, int cnt, char ctype, double lb, double ub, double obj)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}

void GUROBIModel::addRow(const std::string &name, const int *idx, const double *val, int cnt, char sense, double rhs, double rngval)
{
    FP_ASSERT(env && prob);
    char *rname = (char *)(name.c_str());
    int nonconst_idx;

    int *idx_tmp = (int *)malloc(cnt * sizeof(int));
    double *val_tmp = (double *)malloc(cnt * sizeof(double));
    memcpy(idx_tmp, idx, cnt * sizeof(int));
    memcpy(val_tmp, val, cnt * sizeof(double));

    switch (sense)
    {
    case 'G':
        GUROBI_CALL(GRBaddconstr, GRBgetenv(prob), prob, cnt, idx_tmp, val_tmp, GRB_GREATER_EQUAL, rhs, rname);
        break;
    case 'L':
        GUROBI_CALL(GRBaddconstr, GRBgetenv(prob), prob, cnt, idx_tmp, val_tmp, GRB_LESS_EQUAL, rhs, rname);
        break;
    case 'E':
        GUROBI_CALL(GRBaddconstr, GRBgetenv(prob), prob, cnt, idx_tmp, val_tmp, GRB_EQUAL, rhs, rname);
        break;
    default:
        throw std::runtime_error("Unknown sense of linear constrains");
    }
    // GUROBI_CALL(GRBupdatemodel, GRBgetenv(prob), prob);
}

void GUROBIModel::delRow(int ridx)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
void GUROBIModel::delCol(int cidx)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
void GUROBIModel::delRows(int first, int last)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
void GUROBIModel::delCols(int first, int last)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
void GUROBIModel::objSense(ObjSense objsen)
{
    FP_ASSERT(env && prob);
    GUROBI_CALL(GRBsetintattr, GRBgetenv(prob), prob, "ModelSense", static_cast<int>(objsen));
}
void GUROBIModel::objOffset(double val)
{
    FP_ASSERT(env && prob);
    GUROBI_CALL(GRBsetdblattr, GRBgetenv(prob), prob, "ObjCon", val);
    GUROBI_CALL(GRBupdatemodel, GRBgetenv(prob), prob);
}
void GUROBIModel::lb(int cidx, double val)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
void GUROBIModel::lbs(int cnt, const int *cols, const double *values)
{
    FP_ASSERT(env && prob);
    double *values_tmp = (double *)malloc(cnt * sizeof(double));
    memcpy(values_tmp, values, cnt * sizeof(double));
    /*Attention: retrieves a contiguous set of values (cnt values, starting from cols[0] in our example)*/
    GUROBI_CALL(GRBsetdblattrarray, GRBgetenv(prob), prob, "LB", cols[0], cnt, values_tmp);
}
void GUROBIModel::ub(int cidx, double val)
{
    FP_ASSERT(env && prob);
}
void GUROBIModel::ubs(int cnt, const int *cols, const double *values)
{
    FP_ASSERT(env && prob);
    double *values_tmp = (double *)malloc(cnt * sizeof(double));
    memcpy(values_tmp, values, cnt * sizeof(double));
    /*Attention: retrieves a contiguous set of values (cnt values, starting from cols[0] in our example)*/
    GUROBI_CALL(GRBsetdblattrarray, GRBgetenv(prob), prob, "UB", cols[0], cnt, values_tmp);
}

void GUROBIModel::fixCol(int cidx, double val)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}

void GUROBIModel::objcoef(int cidx, double val)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}

void GUROBIModel::objcoefs(int cnt, const int *cols, const double *values)
{
    FP_ASSERT(env && prob);
    double *values_tmp = (double *)malloc(cnt * sizeof(double));
    memcpy(values_tmp, values, cnt * sizeof(double));
    GUROBI_CALL(GRBsetdblattrarray, GRBgetenv(prob), prob, "Obj", cols[0], cnt, values_tmp);
}
void GUROBIModel::ctype(int cidx, char val)
{
    FP_ASSERT(env && prob);
    /* not implemented yet*/
}
/* mark as necessary */
void GUROBIModel::ctypes(int cnt, const int *cols, const char *values)
{
    FP_ASSERT(env && prob);
    // char* values_tmp = (char*) malloc(cnt* sizeof(char));
    // memcpy(values_tmp, values, cnt*sizeof(char));
    // GUROBI_CALL(GRBsetcharattrarray, GRBgetenv(prob), prob, "VType", 0, cnt, values);
}

void GUROBIModel::switchToLP()
{
    FP_ASSERT(env && prob);
    int const count = ncols();
    std::vector<char> ctypesval(count);
    for (int i = 0; i < count; i++)
        ctypesval[i] = GRB_CONTINUOUS;
    GUROBI_CALL(GRBsetcharattrarray, GRBgetenv(prob), prob, "VType", 0, count, &ctypesval[0]);

    // ctypes(count, nullptr, &ctypesval[0]);
}

/* private  */
GUROBIModel *GUROBIModel::clone_impl() const
{
    FP_ASSERT(prob);
    int errcode = 0;

    GRBenv *clonedenv = nullptr;

    /* create an env */
    errcode = GRBloadenv(&clonedenv, NULL);
    if (errcode)
        throwGurobiError(clonedenv);

    /* in case of unstaged changes in prob  */
    GRBupdatemodel(prob);
    GRBmodel *cloned = GRBcopymodel(prob);

    std::unique_ptr<GUROBIModel> cloned_gurobimodel = std::make_unique<GUROBIModel>(clonedenv, cloned);
    return cloned_gurobimodel.release();
}

GUROBIModel *GUROBIModel::presolvedmodel_impl()
{
    FP_ASSERT(prob);
    FP_ASSERT(presolvedProb);
    int errcode = 0;

    GRBenv *clonedenv = nullptr;

    /* create an env */
    errcode = GRBloadenv(&clonedenv, NULL);
    if (errcode)
        throwGurobiError(clonedenv);

    /* in case of unstaged changes in presolvedProb  */
    GRBupdatemodel(presolvedProb);
    GRBmodel *cloned = GRBcopymodel(presolvedProb);

    std::unique_ptr<GUROBIModel> cloned_gurobimodel = std::make_unique<GUROBIModel>(clonedenv, cloned);
    return cloned_gurobimodel.release();
}

#ifdef HAS_XPRESS
MIPModelPtr GUROBIModel::convertToXPRESS()
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
MIPModelPtr GUROBIModel::convertToXPRESS()
{
    assert(false && "XPRESS has not been linked");
    return nullptr;
};
#endif

#ifdef HAS_COPT
MIPModelPtr GUROBIModel::convertToCOPT()
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
MIPModelPtr GUROBIModel::convertToCOPT()
{
    assert(false && "COPT has not been linked");
    return nullptr;
};
#endif

#ifdef HAS_CPLEX
MIPModelPtr GUROBIModel::convertToCPLEX()
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

    /* create the CPLEX problem */
    MIPModelPtr cplex_model = std::make_shared<CPXModel>();

    /* Copy rows and columns (including objective vlaues). */
    for (int j_col = 0; j_col < n_col; ++j_col)
    {
        if ((xtype[j_col] != 'C') && (xtype[j_col] != 'I') && (xtype[j_col] != 'B'))
            throw std::runtime_error("Unsupported variable type for FP");

        cplex_model->addEmptyCol(col_names[j_col], xtype[j_col], xlb[j_col], xub[j_col], xobj[j_col]);
    }

    for (int i_row = 0; i_row < m_row; ++i_row)
    {
        const int nnz = matrix.cnt[i_row];
        const int row_start = matrix.beg[i_row];
        const int *row_idx = &matrix.ind[row_start];
        const double *row_val = &matrix.val[row_start];

        cplex_model->addRow(row_names[i_row], row_idx, row_val, nnz, row_sense[i_row], row_rhs[i_row], rngval[i_row]);
    }
    /* Copy the objective offset. */
    double objoff = this->objOffset();
    cplex_model->objOffset(objoff);

    return cplex_model;
}
#else
MIPModelPtr GUROBIModel::convertToCPLEX()
{
    assert(false && "Cplex has not been linked");
    return nullptr;
};
#endif

MIPModelPtr GUROBIModel::convertTo(const std::string &solver)
{
    if (solver == "GUROBI")
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
    else if (solver == "CPLEX")
        return convertToCPLEX();
#endif

    else
        assert(false && "Not implemented!");

    return nullptr;
}