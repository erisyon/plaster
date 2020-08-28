import pandas as pd
import sys
import time
cimport c_survey_v2_fast as csurvey
cimport c_common as c
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from plaster.tools.log.log import important


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == dtype, f"{arr.dtype} {dtype}"
    assert arr.flags["C_CONTIGUOUS"]


global_progress_callback = None


cdef void _progress(int complete, int total, int retry):
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


IsolationNPType = np.float32


# Wrapper for survey that prepares buffers for csurvey
def survey(
    _n_peps,
    dyemat,
    dyepeps,
    n_threads=1,
    progress=None,
):
    # Views
    cdef c.DyeType [:, ::1] dyemat_view
    cdef c.Index [:, ::1] dyepeps_view
    cdef c.Index [::1] pep_i_to_dyepep_row_i_view
    cdef c.Index [::1] dyt_i_to_mlpep_i_view
    cdef c.Index [::1] dyt_i_to_n_reads_view
    cdef c.IsolationType [::1] pep_i_to_isolation_metric_view
    cdef c.Index [::1] pep_i_to_mic_pep_i_view  # mic = "Most In Contention"

    # Vars
    cdef c.Index n_peps = <c.Index>_n_peps
    cdef c.Size n_dyts = <c.Size>dyemat.shape[0]
    cdef csurvey.SurveyV2FastContext ctx

    cdef int pep_column_in_dyepeps = 1

    assert c.sanity_check() == 0

    global global_progress_callback
    global_progress_callback = progress

    # SETUP the dyemat table
    _assert_array_contiguous(dyemat, np.uint8)
    dyemat_view = dyemat
    ctx.dyemat = c.tab_by_size(&dyemat_view[0, 0], dyemat.nbytes, dyemat.shape[1] * sizeof(c.DyeType), c.TAB_NOT_GROWABLE)

    # BUILD a LUT from dyt_i to most-likely peptide i (mlpep_i)
    # The dyepep_df can have missing pep_i (there are peptides that have no dyt_i)
    # But all dyt have peps.
    dyepep_df = pd.DataFrame(dyepeps, columns=["dyt_i", "pep_i", "n_reads"])

    # EXTRACT the row in each dyt_i group that has the most reads; this is the Most-Likely-Pep
    dyt_i_to_mlpep_i = dyepep_df.loc[dyepep_df.groupby(["dyt_i"])["n_reads"].idxmax()].reset_index()
    assert np.unique(dyt_i_to_mlpep_i.dyt_i).tolist() == list(range(n_dyts))

    dyt_i_to_mlpep_i = dyt_i_to_mlpep_i.pep_i.values
    assert <c.Size>(len(dyt_i_to_mlpep_i)) == n_dyts
    dyt_i_to_mlpep_i = np.ascontiguousarray(dyt_i_to_mlpep_i, dtype=np.uint64)
    _assert_array_contiguous(dyt_i_to_mlpep_i, np.uint64)
    dyt_i_to_mlpep_i_view = dyt_i_to_mlpep_i
    ctx.dyt_i_to_mlpep_i = c.tab_by_size(&dyt_i_to_mlpep_i_view[0], dyt_i_to_mlpep_i.nbytes, sizeof(c.Index), c.TAB_NOT_GROWABLE)

    # FILL-in missing pep_i from the dataframe
    # This is tricky because there can be duplicate "pep_i" rows and the simple reindex
    # answer from SO doesn't work in that case so we need to make a list of the missing rows
    new_index = pd.Index(np.arange(_n_peps), name="pep_i")

    # Drop duplicates from dyepep_df so that the reindex can work...
    missing = dyepep_df.drop_duplicates("pep_i").set_index("pep_i").reindex(new_index)

    # Now missing has all rows, and the "new" rows (ie those that were missing in dyepep_df)
    # have NaNs in their dyt_i fields, so select those out.
    missing = missing[np.isnan(missing.dyt_i)].reset_index()

    # Now we can merge those missing rows into the dyepep_df
    dyepep_df = pd.merge(dyepep_df, missing, on="pep_i", how="outer", suffixes=["", "_dropme"]).drop(columns=["dyt_i_dropme", "n_reads_dropme"])
    dyepep_df = dyepep_df.sort_values(["pep_i", "dyt_i"]).reset_index(drop=True)
    dyepep_df = dyepep_df.fillna(0).astype(np.uint64)

    # SETUP the dyt_i_to_n_reads
    assert np.unique(dyepep_df.dyt_i).tolist() == list(range(n_dyts))
    dyt_i_to_n_reads = np.ascontiguousarray(
        dyepep_df.groupby("dyt_i").sum().reset_index().n_reads.values,
        dtype=np.uint64
    )
    dyt_i_to_n_reads_view = dyt_i_to_n_reads
    ctx.dyt_i_to_n_reads = c.tab_by_size(
        &dyt_i_to_n_reads_view[0],
        dyt_i_to_n_reads.nbytes,
        sizeof(c.Index),
        c.TAB_NOT_GROWABLE
    )

    # SETUP the dyepeps tab, sorting by pep_i.  All pep_i must occur in this.
    assert np.unique(dyepep_df.pep_i).tolist() == list(range(n_peps))
    dyepeps = np.ascontiguousarray(dyepep_df.values, dtype=np.uint64)
    _assert_array_contiguous(dyepeps, np.uint64)
    dyepeps_view = dyepeps
    ctx.dyepeps = c.tab_by_size(
        &dyepeps_view[0, 0],
        dyepeps.nbytes,
        sizeof(c.DyePepRec),
        c.TAB_NOT_GROWABLE
    )

    _pep_i_to_dyepep_row_i = np.unique(dyepep_df.pep_i, return_index=1)[1].astype(np.uint64)
    pep_i_to_dyepep_row_i = np.zeros((n_peps + 1), dtype=np.uint64)
    pep_i_to_dyepep_row_i[0:n_peps] = _pep_i_to_dyepep_row_i
    pep_i_to_dyepep_row_i[n_peps] = dyepeps.shape[0]
    _assert_array_contiguous(pep_i_to_dyepep_row_i, np.uint64)
    pep_i_to_dyepep_row_i_view = pep_i_to_dyepep_row_i
    ctx.pep_i_to_dyepep_row_i = c.tab_by_n_rows(&pep_i_to_dyepep_row_i_view[0], n_peps + 1, sizeof(c.Index), c.TAB_NOT_GROWABLE)

    # SANITY CHECK
    # print(", ".join([f"{i}" for i in pep_i_to_dyepep_row_i.tolist()]))
    assert np.all(np.diff(pep_i_to_dyepep_row_i) >= 0), "bad pep_i_to_dyepep_row_i"

    ctx.n_threads = n_threads
    ctx.n_peps = n_peps
    ctx.n_neighbors = 10
    ctx.n_dyts = n_dyts
    ctx.n_dyt_cols = dyemat.shape[1]
    ctx.distance_to_assign_an_isolated_pep = 10  # TODO: Find this by sampling.
    ctx.progress_fn = <c.ProgressFn>_progress

    pep_i_to_isolation_metric = np.zeros((n_peps,), dtype=IsolationNPType)
    _assert_array_contiguous(pep_i_to_isolation_metric, np.float32)
    pep_i_to_isolation_metric_view = pep_i_to_isolation_metric
    ctx.output_pep_i_to_isolation_metric = c.tab_by_size(
        &pep_i_to_isolation_metric_view[0],
        pep_i_to_isolation_metric.nbytes,
        sizeof(c.IsolationType),
        c.TAB_NOT_GROWABLE
    )

    pep_i_to_mic_pep_i = np.zeros((n_peps,), dtype=np.uint64)
    _assert_array_contiguous(pep_i_to_mic_pep_i, np.uint64)
    pep_i_to_mic_pep_i_view = pep_i_to_mic_pep_i
    ctx.output_pep_i_to_mic_pep_i = c.tab_by_size(
        &pep_i_to_mic_pep_i_view[0],
        pep_i_to_mic_pep_i.nbytes,
        sizeof(c.Index),
        c.TAB_NOT_GROWABLE
    )

    csurvey.context_start(&ctx)

    return pep_i_to_mic_pep_i, pep_i_to_isolation_metric
