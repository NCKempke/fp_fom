#include "gpu_data.h"
#include "cuda_memory.cuh"

GpuModel::GpuModel(const MIPInstance& mip) {
    auto row_matrix = mip.rows;
    auto col_matrix = mip.cols;

    objective = copyinit_to_device(mip.obj);
    lb = copyinit_to_device(mip.lb);
    ub = copyinit_to_device(mip.ub);

    // is_integer = mip.is_integer TODO
    var_type = copyinit_to_device(mip.xtype);

    // bool* is_integer;
    row_val = copyinit_to_device(row_matrix.val);
    col_idx = copyinit_to_device(row_matrix.ind);
    // TODO: is there a sentinel value?
    row_ptr = copyinit_to_device(row_matrix.beg);

    row_val_trans = copyinit_to_device(col_matrix.val);
    col_idx_trans = copyinit_to_device(col_matrix.ind);
    // TODO: is there a sentinel value?
    row_ptr_trans = copyinit_to_device(col_matrix.beg);

    rhs = copyinit_to_device(mip.rhs);
    sense = copyinit_to_device(mip.sense);
    // is_equality = copyinit_to_device(mip.is_equality);

    nrows = mip.nrows;
    ncols = mip.ncols;
}

GpuModel::~GpuModel() {
    device_free(objective);
    device_free(lb);
    device_free(ub);
    device_free(var_type);

    // device_free(is_integer);

    device_free(row_val);
    device_free(col_idx);
    device_free(row_ptr);

    device_free(row_val_trans);
    device_free(col_idx_trans);
    device_free(row_ptr_trans);

    device_free(rhs);
    device_free(sense);
}
