#pragma once

#include "mip.h"
#include "thrust/device_vector.h"

class GpuModelPtrs {
public:
    /* Variables data. */
    const double* objective;
    const double* lb;
    const double* ub;

    /* CSR */
    const double* row_val;
    const int* col_idx;
    const int* row_ptr;

    const double* row_val_trans;
    const int* row_ptr_trans;
    const int* col_idx_trans;

    /* We only allow <= and =  rows. */
    const double* rhs;
};

class GpuModel {
public:
    /* Variables data. */
    thrust::device_vector<double> objective;
    thrust::device_vector<double> lb;
    thrust::device_vector<double> ub;

    /* CSR */
    thrust::device_vector<double> row_val;
    thrust::device_vector<int> col_idx;
    thrust::device_vector<int> row_ptr;

    thrust::device_vector<double> row_val_trans;
    thrust::device_vector<int> row_ptr_trans;
    thrust::device_vector<int> col_idx_trans;

    /* We only allow <= and =  rows. */
    thrust::device_vector<double> rhs;

    /* Rows are sorted [equalities, inequalities]. */
    int nrows;
    int n_equalities;

    /* Columns are sorted [binaries, integers, continuous]. */
    int ncols;
    int n_binaries;
    int n_integers;

    GpuModel(const MIPInstance& data);
    ~GpuModel();

    GpuModelPtrs get_ptrs() const;
};
