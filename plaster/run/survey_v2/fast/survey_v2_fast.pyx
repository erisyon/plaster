import pandas as pd
import sys
import time
cimport c_survey_v2_fast as csurvey
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from plaster.tools.log.log import important


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]


global_progress_callback = None


cdef void _progress(int complete, int total, int retry):
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


IsolationNPType = np.float32


# Wrapper for survey that prepares buffers for csurvey
def survey(
    dyemat,
    dyepeps,
    n_threads=1,
    progress=None,
):
    # Views
    cdef csurvey.DyeType [:, ::1] dyemat_view
    cdef csurvey.Index [:, ::1] dyepeps_view
    cdef csurvey.Index [::1] pep_i_to_dyepep_row_i_view
    cdef csurvey.Index [::1] dyt_i_to_mlpep_i_view
    cdef csurvey.IsolationType [::1] pep_i_to_isolation_metric_view

    # Vars
    cdef csurvey.Index n_peps
    cdef csurvey.Index *pep_i_to_dyepep_row_i_buf
    cdef csurvey.Size n_dyts
    cdef csurvey.Context ctx

    pep_column_in_dyepeps = 1
    n_dyts = dyemat.shape[0]

    global global_progress_callback
    global_progress_callback = progress

    # BUILD a LUT from dyt_i to most-likely peptide i (mlpep_i)
    dyepep_df = pd.DataFrame(dyepeps, columns=["dyt_i", "pep_i", "n_reads"])
    dyt_i_to_mlpep_i = dyepep_df.loc[dyepep_df.groupby(["dyt_i"])["n_reads"].idxmax()].set_index("dyt_i")
    dyt_i_to_mlpep_i = dyt_i_to_mlpep_i.reindex(list(range(0, n_dyts)), fill_value=0)
    dyt_i_to_mlpep_i = dyt_i_to_mlpep_i.pep_i.values
    assert <csurvey.Size>(len(dyt_i_to_mlpep_i)) == n_dyts
    dyt_i_to_mlpep_i = np.ascontiguousarray(dyt_i_to_mlpep_i, dtype=np.uint64)

    _assert_array_contiguous(dyt_i_to_mlpep_i, np.uint64)
    dyt_i_to_mlpep_i_view = dyt_i_to_mlpep_i
    ctx.dyt_i_to_mlpep_i = csurvey.table_init_readonly(<csurvey.Uint8 *>&dyt_i_to_mlpep_i_view[0], dyt_i_to_mlpep_i.nbytes, sizeof(csurvey.Index))

    # SORT by pep_i
    # TODO: This might be quite expensive and unnecsssary?
    dyepeps = dyepeps[dyepeps[:, pep_column_in_dyepeps].argsort()]

    _assert_array_contiguous(dyemat, np.uint8)
    _assert_array_contiguous(dyepeps, np.uint64)
    dyemat_view = dyemat
    dyepeps_view = dyepeps

    ctx.dyemat = csurvey.table_init_readonly(<csurvey.Uint8 *>&dyemat_view[0, 0], dyemat.nbytes, dyemat.shape[1] * sizeof(csurvey.DyeType))
    ctx.dyepeps = csurvey.table_init_readonly(<csurvey.Uint8 *>&dyepeps_view[0, 0], dyepeps.nbytes, sizeof(csurvey.DyePepRec))

    # CREATE an index into the dyepeps records for each peptide block
    # TODO?: DRY with identical logic for pep_i_to_pcb_i in sim_v2
    # Same issue of allocating an extra position so that the last span can be accounted for
    pep_i_to_dyepep_row_i = np.unique(
        dyepeps[:, pep_column_in_dyepeps],
        return_index=1
    )[1].astype(np.uint64)

    _assert_array_contiguous(pep_i_to_dyepep_row_i, np.uint64)
    pep_i_to_dyepep_row_i_view = pep_i_to_dyepep_row_i

    n_peps = pep_i_to_dyepep_row_i.shape[0]
    pep_i_to_dyepep_row_i_buf = <csurvey.Index *>calloc(n_peps + 1, sizeof(csurvey.Index))  # Why +1? see above
    try:
        memcpy(pep_i_to_dyepep_row_i_buf, <const void *>&pep_i_to_dyepep_row_i_view[0], sizeof(csurvey.Index) * n_peps);
        pep_i_to_dyepep_row_i_buf[n_peps] = dyepeps.shape[0]
        ctx.pep_i_to_dyepep_row_i = csurvey.table_init_readonly(<csurvey.Uint8 *>pep_i_to_dyepep_row_i_buf, (n_peps + 1) * sizeof(csurvey.Index), sizeof(csurvey.Index));

        ctx.n_threads = n_threads
        ctx.n_peps = n_peps
        ctx.n_neighbors = 20
        ctx.n_dyts = n_dyts
        ctx.n_dyt_cols = dyemat.shape[1]
        print(f"ctx.n_dyt_cols = {ctx.n_dyt_cols}")
        ctx.distance_to_assign_an_isolated_pep = 10 # TODO: Find this by sampling.
        ctx.progress_fn = <csurvey.ProgressFn>_progress

        pep_i_to_isolation_metric = np.zeros((n_peps,), dtype=IsolationNPType)
        _assert_array_contiguous(pep_i_to_isolation_metric, np.float32)
        pep_i_to_isolation_metric_view = pep_i_to_isolation_metric
        ctx.output_pep_i_to_isolation_metric = csurvey.table_init(<csurvey.Uint8 *>&pep_i_to_isolation_metric_view[0], pep_i_to_isolation_metric.nbytes, sizeof(csurvey.IsolationType))

        csurvey.context_start(&ctx)

        return pep_i_to_isolation_metric
    finally:
        free(pep_i_to_dyepep_row_i_buf)
