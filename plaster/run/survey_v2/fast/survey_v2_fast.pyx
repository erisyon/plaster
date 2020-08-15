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

def tab_tests():
    csurvey.tab_tests()


# Wrapper for survey that prepares buffers for csurvey
def survey(
    _n_peps,
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
    cdef csurvey.Index n_peps = <csurvey.Index>_n_peps
    cdef csurvey.Index n_dyepep_rows = <csurvey.Index>dyepeps.shape[0]
    cdef csurvey.Size n_dyts = <csurvey.Size>dyemat.shape[0]
    cdef csurvey.Context ctx

    cdef int pep_column_in_dyepeps = 1

    global global_progress_callback
    global_progress_callback = progress

    # SETUP the dyemat table
    _assert_array_contiguous(dyemat, np.uint8)
    dyemat_view = dyemat
    ctx.dyemat = csurvey.table_init_readonly(<csurvey.Uint8 *>&dyemat_view[0, 0], dyemat.nbytes, dyemat.shape[1] * sizeof(csurvey.DyeType))

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

    # SETUP the dyepeps table, sorting by pep_i
    # Note, all pep_i must occur in this.
    dyepeps = dyepeps[dyepeps[:, pep_column_in_dyepeps].argsort()]
    pep_i_column = dyepeps[:, pep_column_in_dyepeps]
    if np.unique(pep_i_column).tolist() != list(range(n_peps)):
        print(np.unique(pep_i_column).tolist())
        print(list(range(n_peps)))
    assert np.unique(pep_i_column).tolist() == list(range(n_peps))
    _assert_array_contiguous(dyepeps, np.uint64)
    dyepeps_view = dyepeps
    ctx.dyepeps = csurvey.table_init_readonly(<csurvey.Uint8 *>&dyepeps_view[0, 0], dyepeps.nbytes, sizeof(csurvey.DyePepRec))

    _pep_i_to_dyepep_row_i = np.unique(pep_i_column, return_index=1)[1].astype(np.uint64)
    pep_i_to_dyepep_row_i = np.zeros((n_peps + 1), dtype=np.uint64)
    pep_i_to_dyepep_row_i[0:n_peps] = _pep_i_to_dyepep_row_i
    pep_i_to_dyepep_row_i[n_peps] = n_dyepep_rows
    _assert_array_contiguous(pep_i_to_dyepep_row_i, np.uint64)
    pep_i_to_dyepep_row_i_view = pep_i_to_dyepep_row_i
    ctx.pep_i_to_dyepep_row_i = csurvey.table_init_readonly(
        <csurvey.Uint8 *>&pep_i_to_dyepep_row_i_view[0],
        (n_peps + 1) * sizeof(csurvey.Index),
        sizeof(csurvey.Index)
    );

    # for i in range(n_peps + 1):
    #     print(f"i={i} {pep_i_to_dyepep_row_i_view[i]}")

    ctx.n_threads = n_threads
    ctx.n_peps = n_peps
    ctx.n_neighbors = 20
    ctx.n_dyts = n_dyts
    ctx.n_dyt_cols = dyemat.shape[1]
    ctx.distance_to_assign_an_isolated_pep = 10 # TODO: Find this by sampling.
    ctx.progress_fn = <csurvey.ProgressFn>_progress

    pep_i_to_isolation_metric = np.zeros((n_peps,), dtype=IsolationNPType)
    _assert_array_contiguous(pep_i_to_isolation_metric, np.float32)
    pep_i_to_isolation_metric_view = pep_i_to_isolation_metric
    ctx.output_pep_i_to_isolation_metric = csurvey.table_init_readonly(<csurvey.Uint8 *>&pep_i_to_isolation_metric_view[0], pep_i_to_isolation_metric.nbytes, sizeof(csurvey.IsolationType))

    csurvey.context_start(&ctx)

    return pep_i_to_isolation_metric
