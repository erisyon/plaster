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
    cdef c_nn.Index32 [:, :, ::1] dyepep_view
    cdef c_nn.Index32 [::1] pred_iz_view
    cdef c_nn.Score [::1] scores_view

    radmat_shape = sim_v2_result.test_radmat.shape
    assert len(radmat_shape) == 4
    n_radmat_rows = radmat_shape[0] * radmat_shape[1]
    n_cols = radmat_shape[2] * radmat_shape[3]
    dyemat_shape = sim_v2_result.train_dyemat.shape
    assert len(dyemat_shape) == 4
    n_dyemat_rows = dyemat_shape[0] * dyemat_shape[1]

    radmat_view = sim_v2_result.test_radmat
    dyemat_view = sim_v2_result.train_dyemat
    dyepep_view = sim_v2_result.train_dyepeps_df.values
    n_dyepep_rows = sim_v2_result.train_dyepeps_df.values.shape[0]

    pred_iz = np.zeros((n_radmat_rows,), dtype=np.uint32)
    pred_iz_view = pred_iz
    scores = np.zeros((n_radmat_rows,), dtype=np.float32)
    scores_view = scores

    _assert_array_contiguous(sim_v2_result.test_radmat, np.float32)
    _assert_array_contiguous(sim_v2_result.train_dyemat, np.uint8)
    _assert_array_contiguous(sim_v2_result.train_dyepeps_df.values, np.uint8)

    cdef c_nn.Context ctx
    ctx.n_neighbors = test_nn_params.n_neighbors
    ctx.n_cols = n_cols
    ctx.radmat_n_rows = n_radmat_rows
    ctx.radmat = <c_nn.RadType *>&radmat_view[0, 0]
    ctx.train_dyetracks_n_rows = n_dyemat_rows
    ctx.train_dyetracks = <c_nn.DyeType *>&dyemat_view[0, 0]
    ctx.train_dyepeps_n_rows = n_dyepep_rows
    ctx.train_dyepeps = <c_nn.DyePepRec *>&dyepep_view[0, 0, 0]
    ctx.output_pred_iz = <c_nn.Index32 *>&pred_iz_view[0]
    ctx.output_scores = <c_nn.Score *>&scores_view[0]


    c_nn.context_start(&ctx)
