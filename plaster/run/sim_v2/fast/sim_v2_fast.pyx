import pandas as pd
import sys
import time
cimport c_sim_v2_fast as csim
cimport c_common as c
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from plaster.tools.log.log import important
from cpython.exc cimport PyErr_CheckSignals


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == dtype, f"{arr.dtype} {dtype}"
    assert arr.flags["C_CONTIGUOUS"]


# TODO: Move to c_common
# Type and constant exports
DyeType = np.uint8
CycleKindType = np.uint8
Size = np.uint64
PIType = np.uint64
RecallType = np.float64
PCBType = np.float64
NO_LABEL = c.NO_LABEL
CYCLE_TYPE_PRE = c.CYCLE_TYPE_PRE
CYCLE_TYPE_MOCK = c.CYCLE_TYPE_MOCK
CYCLE_TYPE_EDMAN = c.CYCLE_TYPE_EDMAN


def prob_to_p_i(prob):
    return csim.prob_to_p_i(prob)


global_progress_callback = None

cdef void _progress(int complete, int total, int retry):
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


cdef int _check_keyboard_interrupt_callback():
    try:
        PyErr_CheckSignals()
    except KeyboardInterrupt:
        return 1
    return 0


def max_counts_from_n_peps(n_peps):
    """
    See https://docs.google.com/spreadsheets/d/1GIuox8Rm5H6V3HbazYC713w0grnPSHgEsDD7iH0PwS0/edit#gid=0

    Based on experiments using the count_only option
    I found that n_dyts and n_max_dyepeps grow linearly w/ n_peps

    After some fidding and fiddling I think the following

    So, for 5 channels, 15 cycles, 750_000 peptides:
      Dyts = (8 + 8 + 5 * 15) = 91 * 250 * 750_000 = 17_062_500_000 = 17GB
      DyePepRec = (8 + 8 + 8) = 24 * 450 * 750_000 = 8_100_000_000 = 8GB
      Total = 25 GB

    So, that's a lot, but that's an extreme case...
    I could bring it down in several ways:
    I could store all as 32-bit which would make it:
      Dyts = (4 + 4 + 5 * 15) = 91 * 250 * 750_000 = 15_562_500_000 = 15GB
      DyePepRec = (4 + 4 + 4) = 12 * 450 * 750_000 = 4_050_000_000 = 4GB
      Total = 19GB

    Or, I could stochasitcally remove low-count dyecounts
    which would be a sort of garbage collection operation
    which would probably better than half memory but at more compute time.

    For now, a channel counts I'm likely to run I don't think it will be a problem.
    """
    n_max_dyts = 300 * n_peps + 100_000
    n_max_dyepeps = 450 * n_peps + 100_000
    return n_max_dyts, n_max_dyepeps


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

    pcbs: ndarray[n,3] is a tab of derived from a dataframe.values (float64)
    with columns (pep_i, ch_i, p_bright). These have been sorted
    and we can therefore operate on groups.

    Returns:
        dyetracks: ndarray(shape=(n_uniq_dyetracks, n_channels * n_cycles))
        dyepeps: ndarray(shape=(n_dyepep_rows, 3))
            Where:
                n_dyepep_rows is a unique row for each (dyt_i, pep_i)
                3 columns are: dyt_i, pep_i, count
    """
    cdef c.Uint64 i, j, n_chcy
    cdef csim.Dyt dyt
    cdef c.Index dyt_count
    cdef c.DyeType *dyetrack
    cdef c.DyePepRec *dyepeprec
    cdef c.Size n_peps, n_cycles, n_channels, n_samples
    cdef c.Size n_max_dyts, n_max_dyt_hash_recs
    cdef c.Size n_max_dyepeps, n_max_dyepep_hash_recs
    cdef c.Size count_only = 0  # Set to 1 to use the counting mechanisms
    cdef csim.SimV2FastContext ctx

    # Views
    cdef c.Float64 [:, ::1] pcbs_view
    cdef c.Index [::1] pep_i_to_pcb_i_view
    cdef c.RecallType [::1] pep_recalls_view
    cdef c.Uint8 [:, ::1] dyetracks_view
    cdef c.Uint64 [:, ::1] dyepeps_view

    # Checks
    assert c.sanity_check() == 0
    _assert_array_contiguous(cycles, CycleKindType)
    _assert_array_contiguous(pcbs, PCBType)
    assert np.dtype(CycleKindType).itemsize == sizeof(c.CycleKindType)
    assert np.dtype(DyeType).itemsize == sizeof(c.DyeType)

    # BUILD a map from pep_i to pcb_i.
    #   Note, this map needs to be one longer than n_peps so that we
    #   can subtract each offset to get the pcb length for each pep_i
    pep_i_to_pcb_i = np.unique(pcbs[:, 0], return_index=1)[1].astype(np.uint64)
    pep_i_to_pcb_i_view = pep_i_to_pcb_i
    n_peps = pep_i_to_pcb_i.shape[0]

    n_cycles = cycles.shape[0]
    n_channels = _n_channels
    n_samples = _n_samples

    cdef c.Size n_dyt_row_bytes = csim.dyt_n_bytes(n_channels, n_cycles)

    # How many dyetrack records are needed?
    # I need to run some experiments to find out where I don't allocate

    if count_only == 1:
        n_max_dyts = <c.Size>1
        n_max_dyt_hash_recs = 100_000_000
        n_max_dyepeps = 1
        n_max_dyepep_hash_recs = 100_000_000

    else:
        n_max_dyts, n_max_dyepeps = max_counts_from_n_peps(n_peps)

        hash_factor = 1.5
        n_max_dyt_hash_recs = int(hash_factor * n_max_dyts)
        n_max_dyepep_hash_recs = int(hash_factor * n_max_dyepeps)

        dyt_gb = n_max_dyts * n_dyt_row_bytes / 1024**3
        dyepep_gb = n_max_dyepeps * sizeof(c.DyePepRec) / 1024**3
        if dyt_gb + dyepep_gb > 10:
            important(
                f"Warning: sim_v2 buffers consuming more than 10 GB ({dyt_gb + dyepep_gb:4.1f} GB), "
                f"dyt_gb={dyt_gb}, dyepep_gb={dyepep_gb}, n_max_dyts={n_max_dyts}, n_max_dyepeps={n_max_dyepeps}"
            )

    # Memory
    cdef c.Uint8 *dyts_buf = <c.Uint8 *>calloc(n_max_dyts, n_dyt_row_bytes)
    cdef c.Uint8 *dyepeps_buf = <c.Uint8 *>calloc(n_max_dyepeps, sizeof(c.DyePepRec))
    cdef csim.HashRec *dyt_hash_buf = <csim.HashRec *>calloc(n_max_dyt_hash_recs, sizeof(csim.HashRec))
    cdef csim.HashRec *dyepep_hash_buf = <csim.HashRec *>calloc(n_max_dyepep_hash_recs, sizeof(csim.HashRec))
    cdef c.Index *pep_i_to_pcb_i_buf = <c.Index *>calloc(n_peps + 1, sizeof(c.Index))  # Why +1? see above

    global global_progress_callback
    global_progress_callback = progress

    try:
        if rng_seed is None:
            rng_seed = time.time() * 1_000_000
        ctx.rng_seed = <c.Uint64>rng_seed
        ctx.n_threads = n_threads
        ctx.n_peps = n_peps
        ctx.n_cycles = cycles.shape[0]
        ctx.n_samples = n_samples
        ctx.n_channels = n_channels
        ctx.count_only = count_only
        ctx.pi_bleach = csim.prob_to_p_i(p_bleach)
        ctx.pi_detach = csim.prob_to_p_i(p_detach)
        ctx.pi_edman_success = csim.prob_to_p_i(1.0 - p_edman_fail)
        for i in range(ctx.n_cycles):
            ctx.cycles[i] = cycles[i]

        pcbs_view = pcbs
        ctx.pcbs = c.tab_by_size(<c.Uint8 *>&pcbs_view[0, 0], pcbs.nbytes, sizeof(csim.PCB), c.TAB_NOT_GROWABLE)

        memcpy(pep_i_to_pcb_i_buf, <const void *>&pep_i_to_pcb_i_view[0], sizeof(c.Index) * n_peps);
        pep_i_to_pcb_i_buf[n_peps] = pcbs.shape[0]
        ctx.pep_i_to_pcb_i = c.tab_by_n_rows(<c.Uint8 *>pep_i_to_pcb_i_buf, n_peps + 1, sizeof(c.Index), c.TAB_NOT_GROWABLE);

        ctx.progress_fn = <c.ProgressFn>_progress
        ctx.check_keyboard_interrupt_fn = <c.CheckKeyboardInterruptFn>_check_keyboard_interrupt_callback

        pep_recalls = np.zeros((ctx.n_peps), dtype=RecallType)
        pep_recalls_view = pep_recalls
        ctx.pep_recalls = &pep_recalls_view[0]

        # See sim.c for table and hash definitions
        ctx.dyts = c.tab_by_n_rows(dyts_buf, n_max_dyts, n_dyt_row_bytes, c.TAB_GROWABLE)
        ctx.dyepeps = c.tab_by_size(dyepeps_buf, n_max_dyepeps * sizeof(c.DyePepRec), sizeof(c.DyePepRec), c.TAB_GROWABLE)
        ctx.dyt_hash = csim.hash_init(dyt_hash_buf, n_max_dyt_hash_recs)
        ctx.dyepep_hash = csim.hash_init(dyepep_hash_buf, n_max_dyepep_hash_recs)

        # Now do the work in the C file
        ret = csim.context_work_orders_start(&ctx)
        if ret != 0:
            raise Exception("Worker ended prematurely")

        if count_only:
            print(f"n_dyts={ctx.output_n_dyts}")
            print(f"n_dyepeps={ctx.output_n_dyepeps}")
            return None, None, None

        # The results are in ctx.dyts and ctx.dyepeps
        # So now allocate the numpy arrays that will be returned
        # to the caller and copy into those arrays from the
        # much larger arrays that were used during the context_work_orders_start()
        n_chcy = ctx.n_channels * ctx.n_cycles
        dyetracks = np.zeros((ctx.dyts.n_rows, n_chcy), dtype=DyeType)

        # We need a special record at 0 for nul so we need to add one here
        dyepeps = np.zeros((ctx.dyepeps.n_rows + 1, 3), dtype=Size)
        _assert_array_contiguous(dyetracks, DyeType)
        _assert_array_contiguous(dyepeps, Size)

        dyetracks_view = dyetracks
        dyepeps_view = dyepeps

        for i in range(ctx.dyts.n_rows):
            dyt_count = csim.context_dyt_get_count(&ctx, i)
            dyetrack = csim.context_dyt_dyetrack(&ctx, i)
            for j in range(n_chcy):
                dyetracks_view[i, j] = dyetrack[j]

        # nul record
        dyepeps_view[0, 0] = 0
        dyepeps_view[0, 1] = 0
        dyepeps_view[0, 2] = 0
        for i in range(ctx.dyepeps.n_rows):
            dyepeprec = csim.context_dyepep(&ctx, i)
            dyepeps_view[i+1, 0] = dyepeprec.dyt_i
            dyepeps_view[i+1, 1] = dyepeprec.pep_i
            dyepeps_view[i+1, 2] = dyepeprec.n_reads

        return dyetracks, dyepeps, pep_recalls

    finally:
        free(dyts_buf)
        free(dyepeps_buf)
        free(dyt_hash_buf)
        free(dyepep_hash_buf)
        free(pep_i_to_pcb_i_buf)
