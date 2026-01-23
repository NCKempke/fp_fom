#pragma once

#include "mip.h"
#include "thrust/device_vector.h"

class GpuModel {
public:
    /* Variables data. */
    double* objective;
    double* lb;
    double* ub;
    char* var_type;
    // TODO:
    thrust::device_vector<bool> is_integer;

    /* CSR */
    double* row_val;
    int* col_idx;
    int* row_ptr;

    double* row_val_trans;
    int* row_ptr_trans;
    int* col_idx_trans;

    /* We only allow <= and =  rows. */
    double* rhs;
    char* sense;
    // TODO:
    thrust::device_vector<bool> is_equality;

    int nrows;
    int ncols;

    GpuModel(const MIPInstance& data);
    ~GpuModel();
};