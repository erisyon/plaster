import sys
import time
cimport c_nn_v2_fast as cnn
cimport c_common as c
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from plaster.tools.schema import check
from cpython.exc cimport PyErr_CheckSignals


# Local helpers
def _assert_array_contiguous(arr, dtype, which):
    if not (isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]):
        raise AssertionError(
            f"array {which} is incorrect: "
            f"is_ndarray: {isinstance(arr, np.ndarray)} "
            f"dtype was {arr.dtype} expected {dtype} "
            f"continguous was {arr.flags['C_CONTIGUOUS']}."
        )


def _assert_with_trace(condition, message):
    """
    Cython assert doesn't produce useful traces
    """
    if not condition:
        raise AssertionError(message)


global_progress_callback = None

cdef void _progress(int complete, int total, int retry):
    # print(f"progress {complete} {total} {retry}", file=sys.stderr)
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


cdef int _check_keyboard_interrupt_callback():
    try:
        PyErr_CheckSignals()
    except KeyboardInterrupt:
        return 1
    return 0


def fast_nn(
    test_unit_radmat,
    train_dyemat,
    train_dyepeps,
    n_neighbors,
    n_threads,
    progress=None,
    run_against_all_dyetracks=False,
):
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
        output_pred_dye_iz: ndarray((test_unit_radmat.shape[0],), dtype=np.uint32)
    """
    cdef c.RadType [:, ::1] test_unit_radmat_view
    cdef c.DyeType [:, ::1] train_dyemat_view
    cdef c.Index [:, ::1] train_dyepeps_view
    cdef c.Index32 [::1] output_pred_pep_iz_view
    cdef c.Index32 [::1] output_pred_dye_iz_view
    cdef c.Score [::1] output_scores_view

    # CHECKS
    assert c.sanity_check() == 0
    _assert_array_contiguous(test_unit_radmat, np.float32, "test_unit_radmat")
    _assert_array_contiguous(train_dyemat, np.uint8, "train_dyemat")
    _assert_array_contiguous(train_dyepeps, np.uint64, "train_dyepeps")
    check.array_t(test_unit_radmat, ndim=2)
    check.array_t(train_dyemat, ndim=2)
    n_rows, n_cols =test_unit_radmat.shape
    _assert_with_trace(test_unit_radmat.shape[1] == train_dyemat.shape[1], "radmat and dyemat have different shapes")
    _assert_with_trace(train_dyepeps.ndim == 2 and train_dyepeps.shape[1] == 3, "train_dyepeps has wrong shape")
    _assert_with_trace(np.all(train_dyemat[0, :] == 0.0), "nul row not found in train_dyemat")

    global global_progress_callback
    global_progress_callback = progress

    # ALLOCATE output arrays
    output_pred_pep_iz = np.zeros((test_unit_radmat.shape[0],), dtype=np.uint32)
    output_pred_dye_iz = np.zeros((test_unit_radmat.shape[0],), dtype=np.uint32)
    output_scores = np.zeros((test_unit_radmat.shape[0],), dtype=np.float32)

    # CREATE cython views
    test_unit_radmat_view = test_unit_radmat
    train_dyemat_view = train_dyemat
    output_pred_pep_iz_view = output_pred_pep_iz
    output_pred_dye_iz_view = output_pred_dye_iz
    output_scores_view = output_scores

    cdef c.Index i
    cdef c.Index dye_i
    cdef c.Index last_dye_i
    cdef c.Size last_count
    cdef c.Size n_dyetracks
    cdef c.DyeType *src
    cdef c.RadType *dst
    cdef c.Size train_dyemat_n_rows = train_dyemat.shape[0]
    cdef c.Size train_dyemat_n_elems = train_dyemat_n_rows * n_cols
    cdef c.Size dyepep_n_rows = train_dyepeps.shape[0]
    cdef c.RadType *train_dyemat_as_radtype
    cdef c.Uint64 *dyetrack_weights_uint64
    cdef c.WeightType *dyetrack_weights_float
    cdef c.Index *dye_i_to_dyepep_offset
    cdef cnn.NNV2FastContext ctx

    # COUNT dyetracks (ie, largest dyetrack index + 1)
    train_dyepeps_view = train_dyepeps
    n_dyetracks = 0
    for i in range(dyepep_n_rows):
        n_dyetracks = max(train_dyepeps_view[i, 0], n_dyetracks)
    n_dyetracks += 1  # Because we want a count not max.

    # ALLOCATE arrays
    train_dyemat_as_radtype = <c.RadType *>calloc(train_dyemat_n_elems, sizeof(c.RadType))
    dyetrack_weights_uint64 = <c.Uint64 *>calloc(train_dyemat_n_rows, sizeof(c.Uint64))
    dyetrack_weights_float = <c.WeightType *>calloc(train_dyemat_n_rows, sizeof(c.WeightType))
    dye_i_to_dyepep_offset = <c.Index *>calloc(n_dyetracks, sizeof(c.Index))

    try:
        # CONVERT train_dyemat to floats
        src = <c.DyeType *>&train_dyemat_view[0, 0]
        dst = train_dyemat_as_radtype
        for i in range(train_dyemat_n_elems):
            dst[i] = <c.RadType>src[i]

        # SUM dyepep counts to get weights. Sum to Uint64 then downcast to Float32 to maintain precision
        # CREATE a lookup table to the start of each dyetrack

        # Set last dye to a huge number so that it will be different on first
        # comparison with dye_i below.
        last_dye_i = 0xFFFFFFFFFFFFFFFF
        last_count = 0
        for i in range(dyepep_n_rows):
            dye_i = train_dyepeps_view[i, 0]
            _assert_with_trace(0 <= dye_i < train_dyemat_n_rows, "Bad dye_i index")
            dyetrack_weights_uint64[dye_i] += train_dyepeps_view[i, 2]

            # print(f"dye_i={dye_i} pep_i={train_dyepeps_view[i, 1]} count={train_dyepeps_view[i, 2]} last_dye_i={last_dye_i}")
            if dye_i != last_dye_i:
                _assert_with_trace(dye_i == last_dye_i + 1 or last_dye_i == 0xFFFFFFFFFFFFFFFF, "Non sequential dye_i")
                _assert_with_trace(0 <= dye_i < n_dyetracks, f"Illegal dye_i {dye_i} when setting dye_i_to_dyepep_offset")
                dye_i_to_dyepep_offset[dye_i] = i
            else:
                # Ensure that this is sorted allows picking
                # the most likely pep without a search
                _assert_with_trace(train_dyepeps_view[i, 2] <= last_count, "train_dyepeps_view not sorted")

            last_dye_i = dye_i
            last_count = train_dyepeps_view[i, 2]
            PyErr_CheckSignals()

        # DOWNCAST weights to Float32
        for i in range(train_dyemat_n_rows):
            dyetrack_weights_float[i] = <c.WeightType>dyetrack_weights_uint64[i]

        # SETUP ctx
        ctx.n_neighbors = n_neighbors
        ctx.n_cols = n_cols
        ctx.progress_fn = <c.ProgressFn>_progress
        ctx.check_keyboard_interrupt_fn = <c.CheckKeyboardInterruptFn>_check_keyboard_interrupt_callback

        ctx.test_unit_radmat = c.tab_by_size(
            <c.Uint8 *>&test_unit_radmat_view[0, 0],
            test_unit_radmat.nbytes,
            test_unit_radmat.itemsize * test_unit_radmat.shape[1],
            0
        )

        ctx.train_dyemat = c.tab_by_size(
            <c.Uint8 *>&train_dyemat_as_radtype[0],
            train_dyemat_n_elems * sizeof(c.RadType),
            n_cols * sizeof(c.RadType),
            0
        )

        ctx.train_dyepeps = c.tab_by_size(
            <c.Uint8 *>&train_dyepeps_view[0, 0],
            dyepep_n_rows * sizeof(c.Index) * 3,
            sizeof(c.Index) * 3,
            0
        )

        ctx.train_dye_i_to_dyepep_offset = c.tab_by_size(
            <c.Uint8 *>&dye_i_to_dyepep_offset[0],
            sizeof(c.Index) * n_dyetracks,
            sizeof(c.Index),
            0
        )

        ctx.output_pred_pep_iz = c.tab_by_size(
            <c.Uint8 *>&output_pred_pep_iz_view[0],
            output_pred_pep_iz.nbytes,
            output_pred_pep_iz.itemsize,
            0
        )

        ctx.output_pred_dye_iz = c.tab_by_size(
            <c.Uint8 *>&output_pred_dye_iz_view[0],
            output_pred_dye_iz.nbytes,
            output_pred_dye_iz.itemsize,
            0
        )

        ctx.output_scores = c.tab_by_size(
            <c.Uint8 *>&output_scores_view[0],
            output_scores.nbytes,
            output_scores.itemsize,
            0
        )

        ctx.train_dyetrack_weights = c.tab_by_size(
            <c.Uint8 *>&dyetrack_weights_float[0],
            train_dyemat_n_rows * sizeof(c.WeightType),
            sizeof(c.WeightType),
            0
        )

        ctx.n_threads = n_threads
        ctx.n_rows = n_rows
        ctx.next_row_i = 0
        ctx.n_rows_per_block = 1024 * 16

        # Handoff to the C code...
        ret = cnn.context_start(&ctx)
        if ret != 0:
            raise Exception("Worker ended prematurely")

    finally:
        free(train_dyemat_as_radtype)
        free(dyetrack_weights_uint64)
        free(dyetrack_weights_float)
        free(dye_i_to_dyepep_offset)

    return output_pred_pep_iz, output_scores, output_pred_dye_iz
