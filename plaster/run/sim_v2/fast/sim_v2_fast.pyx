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
    cdef c.Uint64 i, j, n_chcy
    cdef csim.DTR dtr
    cdef c.Index dtr_count
    cdef c.DyeType *dyetrack
    cdef c.DyePepRec *dyepeprec
    cdef c.Size n_peps, n_cycles, n_channels, n_samples
    cdef c.Size n_max_dtrs, n_max_dtr_hash_recs
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

    cdef c.Size n_dtr_row_bytes = csim.dtr_n_bytes(n_channels, n_cycles)

    # How many dyetrack records are needed?
    # I need to run some experiments to find out where I don't allocate

    if count_only == 1:
        n_max_dtrs = <c.Size>1
        n_max_dtr_hash_recs = 100_000_000
        n_max_dyepeps = 1
        n_max_dyepep_hash_recs = 100_000_000
    else:
        # Based on experiments using the count_only option above
        # I found that n_dtrs and n_max_dyepeps grow linearly w/ n_peps
        # I ran experiments over n_channels @ 5000 samples
        #
        # n_ch    |  n_dtr/pep |  n_dyepeps/pep
        # --------|------------|---------------
        #       1 |          4 |             64
        #       2 |          4 |             64
        #       3 |         16 |             97
        #       4 |         87 |            248
        #       5 |        233 |            425
        #
        # After some fidding and fiddling I think the following

        # So, for 5 channels, 15 cycles, 750_000 peptides:
        #   DTRs = (8 + 8 + 5 * 15) = 91 * 250 * 750_000 = 17_062_500_000 = 17GB
        #   DyePepRec = (8 + 8 + 8) = 24 * 450 * 750_000 = 8_100_000_000 = 8GB
        #   Total = 25 GB
        #
        # So, that's a lot, but that's an extreme case...
        # I could bring it down in several ways:
        # I could store all as 32-bit which would make it:
        #   DTRs = (4 + 4 + 5 * 15) = 91 * 250 * 750_000 = 15_562_500_000 = 15GB
        #   DyePepRec = (4 + 4 + 4) = 12 * 450 * 750_000 = 4_050_000_000 = 4GB
        #   Total = 19GB
        #
        # Or, I could stochasitcally remove low-count dyecounts
        # which would be a sort of garbage collection operation
        # which would probably better than half memory but at more compute time.
        #
        # For now, a channel counts I'm likely to run I don't think it will be a problem.
        #
        # Actually this turns out to be pretty dependent on the form of the
        # labels and others. So fo now I'm coverting it to a constant.
        #n_channels_to_n_max_dtr_per_pep = [0, 8, 8, 16, 100, 250]
        #n_channels_to_n_max_dyepep_per_pep = [0, 100, 100, 100, 250, 425]
        extra_factor = 1.2
        hash_factor = 1.5
        n_max_dtrs = <c.Size>(extra_factor * 250 * n_peps + 1000)
        n_max_dtr_hash_recs = int(hash_factor * n_max_dtrs)
        n_max_dyepeps = <c.Size>(extra_factor * 425 * n_peps + 1000)
        n_max_dyepep_hash_recs = int(hash_factor * n_max_dyepeps)
        dtr_mb = n_max_dtrs * n_dtr_row_bytes / 1024**2
        dyepep_mb = n_max_dyepeps * sizeof(c.DyePepRec) / 1024**2
        if dtr_mb + dyepep_mb > 1000:
            important(f"Warning: sim_v2 buffers consuming more than 1 GB ({dtr_mb + dyepep_mb:4.1f} MB)")

    # Memory
    cdef c.Uint8 *dtrs_buf = <c.Uint8 *>calloc(n_max_dtrs, n_dtr_row_bytes)
    cdef c.Uint8 *dyepeps_buf = <c.Uint8 *>calloc(n_max_dyepeps, sizeof(c.DyePepRec))
    cdef csim.HashRec *dtr_hash_buf = <csim.HashRec *>calloc(n_max_dtr_hash_recs, sizeof(csim.HashRec))
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
        ctx.pcbs = c.table_init_readonly(<c.Uint8 *>&pcbs_view[0, 0], pcbs.nbytes, sizeof(csim.PCB))

        memcpy(pep_i_to_pcb_i_buf, <const void *>&pep_i_to_pcb_i_view[0], sizeof(c.Index) * n_peps);
        pep_i_to_pcb_i_buf[n_peps] = pcbs.shape[0]
        ctx.pep_i_to_pcb_i = c.table_init_readonly(<c.Uint8 *>pep_i_to_pcb_i_buf, (n_peps + 1) * sizeof(c.Index), sizeof(c.Index));

        ctx.progress_fn = <c.ProgressFn>_progress

        pep_recalls = np.zeros((ctx.n_peps), dtype=RecallType)
        pep_recalls_view = pep_recalls
        ctx.pep_recalls = &pep_recalls_view[0]

        # See sim.c for table and hash definitions
        ctx.dtrs = c.table_init(dtrs_buf, n_max_dtrs * n_dtr_row_bytes, n_dtr_row_bytes)
        ctx.dyepeps = c.table_init(dyepeps_buf, n_max_dyepeps * sizeof(c.DyePepRec), sizeof(c.DyePepRec))
        ctx.dtr_hash = csim.hash_init(dtr_hash_buf, n_max_dtr_hash_recs)
        ctx.dyepep_hash = csim.hash_init(dyepep_hash_buf, n_max_dyepep_hash_recs)

        # Now do the work in the C file
        csim.context_work_orders_start(&ctx)

        if count_only:
            print(f"n_dtrs={ctx.output_n_dtrs}")
            print(f"n_dyepeps={ctx.output_n_dyepeps}")
            return None, None, None

        # The results are in ctx.dtrs and ctx.dyepeps
        # So now allocate the numpy arrays that will be returned
        # to the caller and copy into those arrays from the
        # much larger arrays that were used during the context_work_orders_start()
        n_chcy = ctx.n_channels * ctx.n_cycles
        dyetracks = np.zeros((ctx.dtrs.n_rows, n_chcy), dtype=DyeType)

        # We need a special record at 0 for nul so we need to add one here
        dyepeps = np.zeros((ctx.dyepeps.n_rows + 1, 3), dtype=Size)
        _assert_array_contiguous(dyetracks, DyeType)
        _assert_array_contiguous(dyepeps, Size)

        dyetracks_view = dyetracks
        dyepeps_view = dyepeps

        for i in range(ctx.dtrs.n_rows):
            dtr_count = csim.context_dtr_get_count(&ctx, i)
            dyetrack = csim.context_dtr_dyetrack(&ctx, i)
            for j in range(n_chcy):
                dyetracks_view[i, j] = dyetrack[j]

        # nul record
        dyepeps_view[0, 0] = 0
        dyepeps_view[0, 1] = 0
        dyepeps_view[0, 2] = 0
        for i in range(ctx.dyepeps.n_rows):
            dyepeprec = csim.context_dyepep(&ctx, i)
            dyepeps_view[i+1, 0] = dyepeprec.dtr_i
            dyepeps_view[i+1, 1] = dyepeprec.pep_i
            dyepeps_view[i+1, 2] = dyepeprec.n_reads

        return dyetracks, dyepeps, pep_recalls

    finally:
        free(dtrs_buf)
        free(dyepeps_buf)
        free(dtr_hash_buf)
        free(dyepep_hash_buf)
        free(pep_i_to_pcb_i_buf)
