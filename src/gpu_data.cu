#include "gpu_data.cuh"
#include "cuda_memory.cuh"

GpuModel::GpuModel(const MIPInstance& mip) {
    auto row_matrix = mip.rows;
    auto col_matrix = mip.cols;

    objective = mip.obj;

    lb = mip.lb;
    ub = mip.ub;

    var_type = mip.xtype;

#ifndef NDEBUG
    for (int icol = 0; icol < mip.ncols; ++icol) {
        FP_ASSERT_IFF(mip.is_integer[icol], mip.xtype[icol] == 'B' || mip.xtype[icol] == 'I');
    }
#endif

    row_val = row_matrix.val;
    col_idx = row_matrix.ind;

    row_ptr = row_matrix.beg;
    FP_ASSERT(row_matrix.beg.size() == row_matrix.k + 1 && row_matrix.beg[row_matrix.k] == row_matrix.nnz);

    row_val_trans = col_matrix.val;
    col_idx_trans = col_matrix.ind;
    row_ptr_trans = col_matrix.beg;
    FP_ASSERT(col_matrix.beg.size() == col_matrix.k + 1 && col_matrix.beg[col_matrix.k] == col_matrix.nnz);

    rhs = mip.rhs;
    row_sense = mip.sense;

#ifndef NDEBUG
    for (int irow = 0; irow < mip.nrows; ++irow) {
        FP_ASSERT_IFF(mip.is_equality[irow], mip.sense[irow] == 'E');
        FP_ASSERT(mip.sense[irow] == 'E' || mip.sense[irow] == 'L');
    }
#endif

    nrows = mip.nrows;
    ncols = mip.ncols;
}

GpuModel::~GpuModel() {
}
