import numpy as np
from plumbum import local
import ctypes as c
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plaster.run.sigproc_v2.c_gauss2_fitter.build import build
from plaster.tools.c_common.c_common_tools import CException
from plaster.tools.image import imops, coord
from plaster.tools.calibration.psf import Gauss2Params, RegPSF
from plaster.tools.log.log import debug
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import F64Arr


class SubPixelAlignContext(c_common_tools.FixupStructure):
    # fmt: off
    _fixup_fields = [
        ("cy_ims", F64Arr),  # Already 1-pixel aligned

        # Parameters
        ("n_cycles", "Size"),
        ("mea_h", "Size"),
        ("mea_w", "Size"),
        ("slice_h", "Size"),
        ("scale", "Size"),

        # Outputs
        ("out_offsets", F64Arr),

        # Internal fields
        ("_n_slices", "Size"),
        ("_cy0_slices", c.c_void_p),
    ]
    # fmt: on


_lib = None


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    with local.cwd("/erisyon/plaster/plaster/run/sigproc_v2/c_sub_pixel_align"):
        build(
            dst_folder="/erisyon/plaster/plaster/run/sigproc_v2/c_sub_pixel_align",
            c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        )
        lib = c.CDLL("./_sub_pixel_align.so")

    lib.context_init.argtypes = [
        c.POINTER(SubPixelAlignContext),
    ]
    lib.context_init.restype = c.c_char_p

    lib.context_free.argtypes = [
        c.POINTER(SubPixelAlignContext),
    ]

    lib.sub_pixel_align_check.argtypes = []
    lib.sub_pixel_align_check.restype = c.c_char_p

    lib.sub_pixel_align_one_cycle.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        ),  # double *p
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        ),  # double *dst_x
    ]
    lib.sub_pixel_align.restype = c.c_char_p

    _lib = lib
    return lib


@contextmanager
def context(cy_ims, slice_h=11):
    """
    with sub_pixel_align.context(...) as ctx:
        zap.work_orders(do_sub_pixel_align, ...)
    """
    lib = load_lib()

    check.array_t(cy_ims, ndim=3, dtype=np.float64)

    n_cycles, mea_h, mea_w = cy_ims.shape

    pixel_offsets, pixel_aligned_cy_ims = imops.align(cy_ims, return_shifted_ims=True)

    out_offsets = np.zeros((n_cycles,), dtype=np.float64)

    ctx = SubPixelAlignContext(
        cy_ims=np.ascontiguousarray(pixel_aligned_cy_ims, dtype=np.float64),
        n_cycles=n_cycles,
        mea_h=mea_h,
        mea_w=mea_w,
        slice_h=slice_h,
        scale=100,
        out_offsets=np.ascontiguousarray(out_offsets, dtype=np.float64),
    )

    error = lib.context_init(ctx)
    if error is not None:
        raise CException(error)

    try:
        yield ctx
    finally:
        lib.context_free(ctx)


def _do_sub_pixel_align_cycle(cy_i, ctx):
    lib = load_lib()
    error = lib.sub_pixel_align_one_cycle(ctx, cy_i)
    if error is not None:
        raise CException(error)


def sub_pixel_align_cy_ims(cy_ims):
    check.array_t(cy_ims, ndim=3, dtype=np.float64)

    n_cycles = cy_ims.shape[0]
    with context(cy_ims) as ctx:
        zap.arrays(
            _do_sub_pixel_align_cycle,
            dict(cy_i=list(range(1, n_cycles))),
            _process_mode=False,
            _trap_exceptions=False,
            ctx=ctx,
        )

    # TODO: Transpose and repeat


def sub_pixel_align_chcy_ims(chcy_ims):
    """
    At some point we may need to align the channels independently.
    For now, combine channels and pass to sub_pixel_align_cy_ims
    """

    check.array_t(chcy_ims, ndim=4, dtype=np.float64)

    cy_ims = np.sum(chcy_ims, axis=0)
    return sub_pixel_align_cy_ims(cy_ims)
