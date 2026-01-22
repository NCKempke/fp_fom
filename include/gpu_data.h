#pragma once

#include "mip.h"

class GpuModel {
public:
    /* Variables data. */
    double* objective;
    double* lb;
    double* ub;
    char* var_type;
    // TODO:
    // bool* is_integer;

    /* CSR */
    double* row_values;
    int* col_idx;
    int* row_ptr;

    double* row_values_trans;
    int* row_ptr_trans;
    int* col_idx_trans;

    /* We only allow <= and =  rows. */
    double* rhs;
    char* sense;
    // TODO:
    // bool* is_equality;

    int nrows;
    int ncols;

    GpuModel(const MIPInstance& data);
    ~GpuModel();
};