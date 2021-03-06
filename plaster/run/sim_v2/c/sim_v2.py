"""
TODO:
    Currently threading takes place in c. At some point we should refactor such that zap handles the parallelism.
    Then we can strip out all of the ctrl-c handling and all of the locking/threading logic in the c file.
    See nn_v2 for an example of how this should work.
"""
import ctypes as c
import signal
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO

import numpy as np
from plaster.run.sim_v2.c.build import build
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import Hash, Tab
from plaster.tools.log.log import debug
from plaster.tools.utils import utils
from plumbum import local

DyeType = np.uint8
Size = np.uint64
PCBType = np.float64
CycleKindType = np.uint8

lib_folder = local.path("/erisyon/plaster/plaster/run/sim_v2/c")

_lib = None


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib = c.CDLL(lib_folder / "_sim_v2.so")

    # C_COMMON
    lib.sanity_check.argtypes = []
    lib.sanity_check.restype = c.c_int

    # SIM_V2
    lib.dyt_n_bytes.argtypes = [c.c_uint64, c.c_uint64]
    lib.dyt_n_bytes.restype = c.c_uint64

    lib.prob_to_p_i.argtypes = [c.c_double]
    lib.prob_to_p_i.restype = c.c_uint64

    lib.context_work_orders_start.argtypes = [
        c.POINTER(SimV2Context),
    ]
    lib.context_work_orders_start.restype = c.c_int

    lib.context_free.argtypes = [
        c.POINTER(SimV2Context),
    ]

    lib.context_dyt_get_count.argtypes = [c.POINTER(SimV2Context), c.c_uint64]
    lib.context_dyt_get_count.restype = c.c_uint64

    lib.context_dyt_dyetrack.argtypes = [c.POINTER(SimV2Context), c.c_uint64]
    lib.context_dyt_dyetrack.restype = c.POINTER(c.c_uint8)

    lib.context_dyepep.argtypes = [c.POINTER(SimV2Context), c.c_uint64]
    lib.context_dyepep.restype = c.POINTER(DyePepRec)

    _lib = lib
    return lib


global_progress_callback = None


@c.CFUNCTYPE(c.c_voidp, c.c_int, c.c_int, c.c_int)
def progress_fn(complete, total, retry):
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


global_interrupted_while_in_c = False


@c.CFUNCTYPE(c.c_int)
def check_keyboard_interrupt_fn():
    return int(global_interrupted_while_in_c)


@contextmanager
def handle_sigint():
    """
    Handles ctrl-c in a special way, intended to be combined with a callback from c to check the global_interrupted_while_in_c value
    """
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def handler(*args, **kwargs):
        global global_interrupted_while_in_c
        global_interrupted_while_in_c = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


class SimV2Context(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("n_peps", "Size"),
        ("n_cycles", "Size"),
        ("n_samples", "Size"),
        ("n_channels", "Size"),
        ("pi_bleach", "Uint64"),
        ("pi_detach", "Uint64"),
        ("pi_edman_success", "Uint64"),
        ("prevent_edman_cterm", "Uint64"),
        ("cycles", "CycleKindType *"),
        ("dyts", Tab, "Uint8"),
        ("dyt_hash", Hash, "Uint8"),  # TODO: add Hash struct to c_common_tools
        ("dyepeps", Tab, "Uint8"),
        ("dyepep_hash", Hash, "Uint8"),  # TODO: add Hash struct to c_common_tools
        ("pcbs", Tab, "Uint8"),
        ("pep_i_to_pcb_i", Tab, "Uint8"),
        ("pep_recalls", "RecallType *"),
        ("next_pep_i", "Index"),
        ("count_only", "Size"),
        ("output_n_dyts", "Size"),
        ("output_n_dyepeps", "Size"),
        ("n_threads", "Size"),
        ("work_order_lock", "pthread_mutex_t *"),
        ("tab_lock", "pthread_mutex_t *"),
        ("rng_seed", "Uint64"),
        ("progress_fn", "ProgressFn"),
        ("check_keyboard_interrupt_fn", "KeyboardInterruptFn"),
        ("pep_i_to_pcb_i_buf", "Index *"),
        ("n_max_dyts", "Size"),
        ("n_max_dyepeps", "Size"),
        ("n_dyt_row_bytes", "Size"),
        ("n_max_dyt_hash_recs", "Size"),
        ("n_max_dyepep_hash_recs", "Size"),
    ]


# class Dyt(c_common_tools.FixupStructure):
#     _fixup_fields = [
#         ("count", "Size"),
#         ("dyt_i", "Index"),
#         ("chcy_dye_counts", "DyeType *"),
#     ]


class PCB(c_common_tools.FixupStructure):
    _fixup_fields = [("pep_i", "Float64"), ("ch_i", "Float64"), ("p_bright", "Float64")]


class Counts(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("n_new_dyts", "Size"),
        ("n_new_dyepeps", "Size"),
    ]


class DyePepRec(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("dyt_i", "Index"),
        ("pep_i", "Index"),
        ("n_reads", "Size"),
    ]


def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == dtype, f"{arr.dtype} {dtype}"
    assert arr.flags["C_CONTIGUOUS"]


def init():
    """
    Must be called before anything else in this module
    """

    SimV2Context.struct_fixup()
    # Dyt.struct_fixup()
    PCB.struct_fixup()
    Counts.struct_fixup()
    DyePepRec.struct_fixup()

    with local.cwd(lib_folder):
        fp = StringIO()
        with redirect_stdout(fp):
            print(f"// This file was code-generated by sim_v2.c.sim_v2.init")
            print()
            print("#ifndef SIM_V2_H")
            print("#define SIM_V2_H")
            print()
            print('#include "stdint.h"')
            print('#include "c_common.h"')
            print()
            print(
                """
                typedef struct {
                    Size count;
                    Index dyt_i;
                    DyeType chcy_dye_counts[];
                    // Note, this is a variable sized record
                    // See dyt_* functions for manipulating it
                } Dyt;  // Dye-track record
                """
            )
            print()
            SimV2Context.struct_emit_header(fp)
            # Dyt.struct_emit_header(fp)
            PCB.struct_emit_header(fp)
            Counts.struct_emit_header(fp)
            print("#endif")

        header_file_path = "./_sim_v2.h"
        existing_h = utils.load(header_file_path, return_on_non_existing="")

        if existing_h != fp.getvalue():
            utils.save(header_file_path, fp.getvalue())

        build(
            dst_folder=lib_folder,
            c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        )
        lib = c.CDLL("./_sim_v2.so")


def max_counts(n_peps, n_labels, n_channels):
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

    10/28/2020 DHW changed these equations to include n_labels and n_channels. See third sheet in linked doc
    This is potentially slightly better, but there's still another factor besides n_peps, n_labels, and n_channels that needs to be used,
    though I'm not sure what it is yet.
    ZBS: I bet that another factor is the MEAN of the length of the peptides that are generated by
    that protease. You can get the length of a peptide in the "prep_result". There are a variety of dataframe requests and with a little
    panda-kung-fu you should be able to get the mean length
    """
    # It's possible that the max doesn't grow linearlly across n_labels, though I'm not sure.
    # For some reason when n_labels == 1 things act really differently, not sure what that's about
    n_max_dyts = n_peps * n_channels * max(n_labels, 2) ** 1.5 * 50 + 100_000
    n_max_dyepeps = n_peps * n_channels * max(n_labels, 2) ** 1.5 * 90 + 100_000
    return n_max_dyts, n_max_dyepeps


def sim(
    pcbs,  # pcb = (p)ep_i, (c)h_i, (b)right_prob
    n_samples,
    n_channels,
    n_labels,
    cycles,
    p_bleach,
    p_detach,
    p_edman_fail,
    prevent_edman_cterm,
    n_threads=1,
    rng_seed=None,
    progress=None,
):
    count_only = 0  # Set to 1 to use the counting mechanisms

    global global_progress_callback
    global_progress_callback = progress

    lib = load_lib()

    # TODO:
    assert lib.sanity_check() == 0
    _assert_array_contiguous(cycles, CycleKindType)
    _assert_array_contiguous(pcbs, PCBType)

    # BUILD a map from pep_i to pcb_i.
    #   Note, this map needs to be one longer than n_peps so that we
    #   can subtract each offset to get the pcb length for each pep_i
    pep_i_to_pcb_i = np.unique(pcbs[:, 0], return_index=1)[1].astype(np.uint64)
    pep_i_to_pcb_i_view = pep_i_to_pcb_i
    n_peps = pep_i_to_pcb_i.shape[0]

    pep_i_to_pcb_i_buf = (c.c_ulonglong * (n_peps + 1))()
    c.memmove(
        pep_i_to_pcb_i_buf,
        pep_i_to_pcb_i_view.ctypes.data,
        n_peps * c.sizeof(c.c_ulonglong),
    )
    pep_i_to_pcb_i_buf[n_peps] = pcbs.shape[0]

    n_cycles = cycles.shape[0]

    n_dyt_row_bytes = lib.dyt_n_bytes(n_channels, n_cycles)

    # How many dyetrack records are needed?
    # I need to run some experiments to find out where I don't allocate

    if count_only == 1:
        n_max_dyts = 1
        n_max_dyt_hash_recs = 100_000_000
        n_max_dyepeps = 1
        n_max_dyepep_hash_recs = 100_000_000

    else:
        n_max_dyts, n_max_dyepeps = max_counts(n_peps, n_labels, n_channels)

        hash_factor = 1.5
        n_max_dyt_hash_recs = int(hash_factor * n_max_dyts)
        n_max_dyepep_hash_recs = int(hash_factor * n_max_dyepeps)

        dyt_gb = n_max_dyts * n_dyt_row_bytes / 1024 ** 3
        dyepep_gb = n_max_dyepeps * c.sizeof(DyePepRec) / 1024 ** 3
        if dyt_gb + dyepep_gb > 10:
            important(
                f"Warning: sim_v2 buffers consuming more than 10 GB ({dyt_gb + dyepep_gb:4.1f} GB), "
                f"dyt_gb={dyt_gb}, dyepep_gb={dyepep_gb}, n_max_dyts={n_max_dyts}, n_max_dyepeps={n_max_dyepeps}"
            )

    # It's important that we hold onto a reference to this ndarray before we drop into c so it's not GC'd
    pep_recalls = np.zeros(n_peps, dtype=np.float64)

    ctx = SimV2Context(
        n_peps=n_peps,
        n_cycles=n_cycles,
        n_samples=n_samples,
        n_channels=n_channels,
        pi_bleach=lib.prob_to_p_i(p_bleach),
        pi_detach=lib.prob_to_p_i(p_detach),
        pi_edman_success=lib.prob_to_p_i(1.0 - p_edman_fail),
        prevent_edman_cterm=prevent_edman_cterm,
        cycles=(c.c_uint8 * 64)(),
        pcbs=Tab.from_mat(pcbs, expected_dtype=np.float64),
        n_max_dyts=int(n_max_dyts),
        n_max_dyt_hash_recs=int(n_max_dyt_hash_recs),
        n_max_dyepeps=int(n_max_dyepeps),
        n_max_dyepep_hash_recs=int(n_max_dyepep_hash_recs),
        n_dyt_row_bytes=n_dyt_row_bytes,
        # TODO: look at F64Arr
        pep_recalls=pep_recalls.ctypes.data_as(c.POINTER(c.c_double)),
        n_threads=n_threads,
        progress_fn=progress_fn,
        check_keyboard_interrupt_fn=check_keyboard_interrupt_fn,
        rng_seed=int(time.time() * 1_000_000),
        count_only=count_only,
        pep_i_to_pcb_i_buf=pep_i_to_pcb_i_buf,
    )

    for i in range(ctx.n_cycles):
        ctx.cycles[i] = cycles[i]

    try:
        # TODO: use convention in radiometry.py with context_init in a context manager, so ctx is always freed
        with handle_sigint():
            ret = lib.context_work_orders_start(ctx)
        if ret != 0:
            raise Exception(f"Worker ended prematurely {ret}")

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
            dyt_count = lib.context_dyt_get_count(ctx, i)
            dyetrack = lib.context_dyt_dyetrack(ctx, i)
            for j in range(n_chcy):
                dyetracks_view[i, j] = dyetrack[j]

        # nul record
        dyepeps_view[0, 0] = 0
        dyepeps_view[0, 1] = 0
        dyepeps_view[0, 2] = 0
        for i in range(ctx.dyepeps.n_rows):
            dyepeprec = lib.context_dyepep(ctx, i).contents
            dyepeps_view[i + 1, 0] = dyepeprec.dyt_i
            dyepeps_view[i + 1, 1] = dyepeprec.pep_i
            dyepeps_view[i + 1, 2] = dyepeprec.n_reads

        return dyetracks, dyepeps, pep_recalls
    finally:
        lib.context_free(ctx)
