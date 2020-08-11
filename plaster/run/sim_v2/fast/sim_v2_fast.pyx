import sys
import time
cimport c_sim_v2_fast as csim
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]


# Type and constant exports
DyeType = np.uint8
CycleKindType = np.uint8
Size = np.uint64
PIType = np.uint64
RecallType = np.float64
PCBType = np.float64
NO_LABEL = csim.NO_LABEL
CYCLE_TYPE_PRE = csim.CYCLE_TYPE_PRE
CYCLE_TYPE_MOCK = csim.CYCLE_TYPE_MOCK
CYCLE_TYPE_EDMAN = csim.CYCLE_TYPE_EDMAN


def prob_to_p_i(prob):
    return csim.prob_to_p_i(prob)


global_progress_callback = None

cdef void _progress(int complete, int total, int retry):
    # print(f"progress {complete} {total} {retry}", file=sys.stderr)
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


# Wrapper for sim that prepares buffers for csim
def sim(
    pcbs,  # pcb = (p)ep_i, (c)h_i, (b)right_prob
    _n_samples,
    _n_channels,
    cycles,
    p_bleach,
    p_detach,
    p_edman_fail,
    n_threads=1,
    rng_seed=None,
    progress=None,
):
    """
    Run the Virtual Fluoro Sequence Monte-Carlo simulator via Cython to
    the C implemntation in csim_v2_fast.c.

    pcbs: ndarray[n,3] is a table of derived from a dataframe.values (float64)
    with columns (pep_i, ch_i, p_bright). These have been sorted
    and we can therefore operate on groups.

    Returns:
        dyetracks: ndarray(shape=(n_uniq_dyetracks, n_channels * n_cycles))
        dyepeps: ndarray(shape=(n_dyepep_rows, 3))
            Where:
                n_dyepep_rows is a unique row for each (dtr_i, pep_i)
                3 columns are: dtr_i, pep_i, count
    """
    cdef csim.Uint64 i, j, n_chcy
    cdef csim.DTR dtr
    cdef csim.Index dtr_count
    cdef csim.DyeType *dyetrack
    cdef csim.DyePepRec *dyepeprec
    cdef csim.Size n_peps, n_cycles, n_channels, n_samples
    cdef csim.Context ctx

    # Views
    cdef csim.Float64 [:, ::1] pcbs_view
    cdef csim.Index [::1] pep_i_to_pcb_i_view
    cdef csim.RecallType [::1] pep_recalls_view
    cdef csim.Uint8 [:, ::1] dyetracks_view
    cdef csim.Uint64 [:, ::1] dyepeps_view

    # Checks
    _assert_array_contiguous(cycles, CycleKindType)
    _assert_array_contiguous(pcbs, PCBType)
    assert np.dtype(CycleKindType).itemsize == sizeof(csim.CycleKindType)
    assert np.dtype(DyeType).itemsize == sizeof(csim.DyeType)

    # BUILD a map from pep_i to pcb_i.
    #   Note, this map needs to be one longer than n_peps so that we
    #   can subtract each offset to get the pcb length for each pep_i
    pep_i_to_pcb_i = np.unique(pcbs[:, 0], return_index=1)[1].astype(np.uint64)
    pep_i_to_pcb_i_view = pep_i_to_pcb_i
    n_peps = pep_i_to_pcb_i.shape[0]
    n_cycles = cycles.shape[0]
    n_channels = _n_channels
    n_samples = _n_samples

    cdef csim.Size n_dtr_row_bytes = csim.dtr_n_bytes(n_channels, n_cycles)
    cdef csim.Size n_max_dtrs = <csim.Size>(0.8 * n_peps * n_samples)
    cdef csim.Size n_max_dtr_hash_recs = int(1.5 * n_max_dtrs)
    cdef csim.Size n_max_dyepeps = int(1.5 * n_max_dtrs)
    cdef csim.Size n_max_dyepep_hash_recs = int(1.5 * n_max_dyepeps)

    # Memory
    cdef csim.Uint8 *dtrs_buf = <csim.Uint8 *>calloc(n_max_dtrs, n_dtr_row_bytes)
    cdef csim.Uint8 *dyepeps_buf = <csim.Uint8 *>calloc(n_max_dyepeps, sizeof(csim.DyePepRec))
    cdef csim.HashRec *dtr_hash_buf = <csim.HashRec *>calloc(n_max_dtr_hash_recs, sizeof(csim.HashRec))
    cdef csim.HashRec *dyepep_hash_buf = <csim.HashRec *>calloc(n_max_dyepep_hash_recs, sizeof(csim.HashRec))
    cdef csim.Index *pep_i_to_pcb_i_buf = <csim.Index *>calloc(n_peps + 1, sizeof(csim.Index))  # Why +1? see above

    global global_progress_callback
    global_progress_callback = progress

    try:
        if rng_seed is None:
            rng_seed = time.time() * 1_000_000
        ctx.rng_seed = <csim.Uint64>rng_seed
        ctx.n_threads = n_threads
        ctx.n_peps = n_peps
        ctx.n_cycles = cycles.shape[0]
        ctx.n_samples = n_samples
        ctx.n_channels = n_channels
        ctx.pi_bleach = csim.prob_to_p_i(p_bleach)
        ctx.pi_detach = csim.prob_to_p_i(p_detach)
        ctx.pi_edman_success = csim.prob_to_p_i(1.0 - p_edman_fail)
        for i in range(ctx.n_cycles):
            ctx.cycles[i] = cycles[i]

        pcbs_view = pcbs
        ctx.pcbs = csim.table_init_readonly(<csim.Uint8 *>&pcbs_view[0, 0], pcbs.nbytes, sizeof(csim.PCB))

        memcpy(pep_i_to_pcb_i_buf, <const void *>&pep_i_to_pcb_i_view[0], sizeof(csim.Index) * n_peps);
        pep_i_to_pcb_i_buf[n_peps] = pcbs.shape[0]
        ctx.pep_i_to_pcb_i = csim.table_init_readonly(<csim.Uint8 *>pep_i_to_pcb_i_buf, (n_peps + 1) * sizeof(csim.Index), sizeof(csim.Index));

        ctx.progress_fn = <csim.ProgressFn>_progress

        pep_recalls = np.zeros((ctx.n_peps), dtype=RecallType)
        pep_recalls_view = pep_recalls
        ctx.pep_recalls = &pep_recalls_view[0]

        # See sim.c for table and hash definitions
        ctx.dtrs = csim.table_init(dtrs_buf, n_max_dtrs * n_dtr_row_bytes, n_dtr_row_bytes)
        ctx.dyepeps = csim.table_init(dyepeps_buf, n_max_dyepeps * sizeof(csim.DyePepRec), sizeof(csim.DyePepRec))
        ctx.dtr_hash = csim.hash_init(dtr_hash_buf, n_max_dtr_hash_recs)
        ctx.dyepep_hash = csim.hash_init(dyepep_hash_buf, n_max_dyepep_hash_recs)

        # Now do the work in the C file
        csim.context_work_orders_start(&ctx)

        # The results are in ctx.dtrs and ctx.dyepeps
        # So now allocate the numpy arrays that will be returned
        # to the caller and copy into those arrays from the
        # much larger arrays that were used during the context_work_orders_start()
        n_chcy = ctx.n_channels * ctx.n_cycles
        dyetracks = np.zeros((ctx.dtrs.n_rows, n_chcy), dtype=DyeType)
        dyepeps = np.zeros((ctx.dyepeps.n_rows, 3), dtype=Size)
        _assert_array_contiguous(dyetracks, DyeType)
        _assert_array_contiguous(dyepeps, Size)

        dyetracks_view = dyetracks
        dyepeps_view = dyepeps

        for i in range(ctx.dtrs.n_rows):
            dtr_count = csim.context_dtr_get_count(&ctx, i)
            dyetrack = csim.context_dtr_dyetrack(&ctx, i)
            for j in range(n_chcy):
                dyetracks_view[i, j] = dyetrack[j]

        for i in range(ctx.dyepeps.n_rows):
            dyepeprec = csim.context_dyepep(&ctx, i)
            dyepeps_view[i, 0] = dyepeprec.dtr_i
            dyepeps_view[i, 1] = dyepeprec.pep_i
            dyepeps_view[i, 2] = dyepeprec.count

        return dyetracks, dyepeps, pep_recalls

    finally:
        free(dtrs_buf)
        free(dyepeps_buf)
        free(dtr_hash_buf)
        free(dyepep_hash_buf)
        free(pep_i_to_pcb_i_buf)
