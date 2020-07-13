import time
cimport c_nn_v2_fast as c_nn
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]


def nn(test_nn_params, prep_result, sim_v2_result):
    cdef c_nn.DyeType [:, ::1] dyemat_view
    cdef c_nn.RadType [:, ::1] radmat_view
    cdef c_nn.Size [:, ::1] dyepep_view
    cdef c_nn.Index32 [:, ::1] pred_iz_view
    cdef c_nn.Score [:, ::1] scores_view

    radmat_shape = sim_v2_result.test_radmat.shape
    assert len(radmat_shape) == 4
    n_radmat_rows = shape[0] * shape[1]
    n_cols = shape[2] * shape[3]
    dyemat_shape = sim_v2_result.train_dyemat.shape
    assert len(dyemat_shape) == 4
    n_dyemat_rows = shape[0] * shape[1]

    radmat_view = sim_v2_result.test_radmat
    dyemat_view = sim_v2_result.train_dyemat
    dyepep_view = sim_v2_result.train_dyepeps_df.values
    n_dyepep_rows = sim_v2_result.train_dyepeps_df.values.shape[0]

    pred_iz = np.zeros((n_radmat_rows,), dtype=np.uint32)
    pred_iz_view = pred_iz
    scores = np.zeros((n_radmat_rows,), dtype=np.float32)
    scores_view = scores

    _assert_array_contiguous(sim_v2_result.test_radmat, RadType)
    _assert_array_contiguous(sim_v2_result.train_dyemat, DyeType)
    _assert_array_contiguous(sim_v2_result.train_dyepeps_df.values, DyeType)

    cdef c_nn.Context ctx
    ctx.n_cols = n_cols
    ctx.radmat_n_rows = n_rows
    ctx.radmat = radmat_view
    ctx.train_dyetracks_n_rows = n_dyemat_rows
    ctx.train_dyetracks = dyemat_view
    ctx.train_dyepeps_n_rows = n_dyepep_rows
    ctx.train_dyepeps = dyepep_view
    ctx.output_pred_iz = pred_iz_view
    ctx.output_scores = scores_view

    c_nn.context_start(&ctx)
