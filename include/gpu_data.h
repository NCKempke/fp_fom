#pragma once

#include "mip.h"
#include "thrust/device_vector.h"

class GpuModel {
public:
    /* Variables data. */
    thrust::device_vector<double> objective;
    thrust::device_vector<double> lb;
    thrust::device_vector<double> ub;
    thrust::device_vector<char> var_type;

    /* CSR */
    thrust::device_vector<double> row_val;
    thrust::device_vector<int> col_idx;
    thrust::device_vector<int> row_ptr;

    thrust::device_vector<double> row_val_trans;
    thrust::device_vector<int> row_ptr_trans;
    thrust::device_vector<int> col_idx_trans;

    /* We only allow <= and =  rows. */
    thrust::device_vector<double> rhs;
    thrust::device_vector<char> row_sense;

    int nrows;
    int ncols;

    GpuModel(const MIPInstance& data);
    ~GpuModel();
};