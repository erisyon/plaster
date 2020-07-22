import time
cimport c_nn_v2_fast as c_nn
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from plaster.tools.schema import check


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]


def fast_nn(test_unit_radmat, train_dyemat, train_dyepeps, n_neighbors):
    """
    This is the interface to the C implementation of NN.

    Arguments:
        test_unit_radmat: ndarray((n_rows, n_channels * n_cycles), dtype=np.float32)
            The unit_radmat to test. May come from the scope or simulated. Is already
            normalized (1 unit = 1 dye, all channels equalized)
        train_dyemat: ndarray((n_rows, n_channels * n_cycles), dtype=np.uint8)
            The dyemat comes from sim_v2. Equal number of samples per peptide
        train_dyepeps: ndarray((n_rows, 3), dtype=np.uint32)
            Three columns: (dyetrack_i, pep_i, count)
            Where:
                dyetrack_i: is the index into the row of train_dyemat
                pep_i: is a peptide index
                count: is the number of times that pep_i created this dyetrack_i

    Returns:
        (pred_pep_iz, scores)
        pred_pep_iz: ndarray((test_unit_radmat.shape[0],), dtype=np.uint32)
        scores: ndarray((test_unit_radmat.shape[0],), dtype=np.float32)
    """
    cdef c_nn.RadType [:, ::1] test_unit_radmat_view
    cdef c_nn.DyeType [:, ::1] train_dyemat_view
    cdef c_nn.Index [:, ::1] train_dyepeps_view
    cdef c_nn.Index32 [::1] output_pred_iz_view
    cdef c_nn.Score [::1] output_scores_view

    # CHECKS
    _assert_array_contiguous(test_unit_radmat, np.float32)
    _assert_array_contiguous(train_dyemat, np.uint8)
    _assert_array_contiguous(train_dyepeps, np.uint64)
    check.array_t(test_unit_radmat, ndim=2)
    check.array_t(train_dyemat, ndim=2)
    n_cols = test_unit_radmat.shape[1]
    assert test_unit_radmat.shape[1] == train_dyemat.shape[1]
    assert train_dyepeps.ndim == 2 and train_dyepeps.shape[1] == 3

    # ALLOCATE output arrays
    output_pred_iz = np.zeros((test_unit_radmat.shape[0],), dtype=np.uint32)
    output_scores = np.zeros((test_unit_radmat.shape[0],), dtype=np.float32)

    # CREATE cython views
    test_unit_radmat_view = test_unit_radmat
    train_dyemat_view = train_dyemat
    output_pred_iz_view = output_pred_iz
    output_scores_view = output_scores

    cdef c_nn.Index i
    cdef c_nn.Index dt_i
    cdef c_nn.DyeType *src
    cdef c_nn.RadType *dst
    cdef c_nn.Size train_dyemat_n_rows = train_dyemat.shape[0]
    cdef c_nn.Size train_dyemat_n_elems = train_dyemat_n_rows * n_cols
    cdef c_nn.RadType *train_dyemat_as_radtype = <c_nn.RadType *>calloc(train_dyemat_n_elems, sizeof(c_nn.RadType))
    cdef c_nn.Uint64 *dyetrack_weights_uint64 = <c_nn.Uint64 *>calloc(train_dyemat_n_rows, sizeof(c_nn.Uint64))
    cdef c_nn.WeightType *dyetrack_weights_float = <c_nn.WeightType *>calloc(train_dyemat_n_rows, sizeof(c_nn.WeightType))
    cdef c_nn.Size dyepep_n_rows = train_dyepeps.shape[0]
    cdef c_nn.Context ctx

    try:
        # CONVERT train_dyemat to floats
        src = <c_nn.DyeType *>&train_dyemat_view[0, 0]
        dst = train_dyemat_as_radtype
        for i in range(train_dyemat_n_elems):
            dst[i] = <c_nn.RadType>src[i]

        # SUM dyepep counts to get weights. Sum to Uint64 then downcast to Float32 to maintain precision
        train_dyepeps_view = train_dyepeps
        for i in range(dyepep_n_rows):
            dt_i = train_dyepeps_view[i, 0]
            assert 0 <= dt_i < train_dyemat_n_rows
            dyetrack_weights_uint64[dt_i] += train_dyepeps_view[i, 2]

        # DOWNCAST weights to Float32
        for i in range(train_dyemat_n_rows):
            dyetrack_weights_float[i] = <c_nn.WeightType>dyetrack_weights_uint64[i]

        # SETUP ctx
        ctx.n_neighbors = n_neighbors
        ctx.n_cols = n_cols

        ctx.test_unit_radmat = c_nn.table_init_readonly(
            <c_nn.Uint8 *>&test_unit_radmat_view[0, 0],
            test_unit_radmat.nbytes,
            test_unit_radmat.itemsize * test_unit_radmat.shape[1]
        )

        ctx.train_dyemat = c_nn.table_init_readonly(
            <c_nn.Uint8 *>&train_dyemat_as_radtype[0],
            train_dyemat_n_elems * sizeof(c_nn.RadType),
            n_cols * sizeof(c_nn.RadType)
        )

        ctx.output_pred_iz = c_nn.table_init(
            <c_nn.Uint8 *>&output_pred_iz_view[0],
            output_pred_iz.nbytes,
            output_pred_iz.itemsize
        )

        ctx.output_scores = c_nn.table_init(
            <c_nn.Uint8 *>&output_scores_view[0],
            output_scores.nbytes,
            output_scores.itemsize
        )

        ctx.train_dyetrack_weights = c_nn.table_init_readonly(
            <c_nn.Uint8 *>&dyetrack_weights_float[0],
            train_dyemat_n_rows * sizeof(c_nn.WeightType),
            sizeof(c_nn.WeightType)
        )

        # Handoff to the C code...
        c_nn.context_start(&ctx)

    finally:
        free(train_dyemat_as_radtype)
        free(dyetrack_weights_uint64)
        free(dyetrack_weights_float)

    return output_pred_iz, output_scores

