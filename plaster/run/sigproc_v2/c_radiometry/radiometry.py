import numpy as np
from plumbum import local
import ctypes as c
from io import StringIO
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plaster.run.sigproc_v2.c_radiometry.build import build
from plaster.run.sigproc_v2.reg_psf import RegPSF
from plaster.tools.c_common.c_common_tools import CException
from plaster.tools.utils import utils
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import F64Arr
from plaster.tools.log.log import debug


class RadiometryContext(c_common_tools.FixupStructure):
    # Operates on a single field stack so that we don't have
    # to realize all fields in to memory simultaneously

    # fmt: off
    _fixup_fields = [
        ("chcy_ims", F64Arr),  # Sub-pixel aligned  (n_channels, n_cycles, height, width)
        ("locs", F64Arr),  # Sub-pixel centered  (n_peaks, (y, x))
        ("reg_psf_params", F64Arr),  # Reg_psf array (n_divs, n_divs, mea, mea)
        ("focus_adjustment", F64Arr),  # focus adjustment per cycle (n_cycles)

        # Parameters
        ("n_channels", "Size"),
        ("n_cycles", "Size"),
        ("n_peaks", "Size"),
        ("n_divs", "Size"),
        ("peak_mea", "Size"),
        ("height", "Size"),
        ("width", "Size"),

        # Outputs
        ("out_radiometry", F64Arr),  # signal, noise, snr, aspect_ratio
    ]
    # fmt: on


_lib = None


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    RadiometryContext.struct_fixup()

    with local.cwd("/erisyon/plaster/plaster/run/sigproc_v2/c_radiometry"):
        fp = StringIO()
        with redirect_stdout(fp):
            print(
                f"// This file was code-generated by sigproc_v2.c_radiometry.c_radiometry.load_lib"
            )
            print()
            print("#ifndef RADIOMETRY_H")
            print("#define RADIOMETRY_H")
            print()
            print('#include "stdint.h"')
            print('#include "c_common.h"')
            print()
            RadiometryContext.struct_emit_header(fp)
            print("#endif")

        header_file_path = "./_radiometry.h"
        existing_h = utils.load(header_file_path, return_on_non_existing="")

        if existing_h != fp.getvalue():
            utils.save(header_file_path, fp.getvalue())

        build(
            dst_folder="/erisyon/plaster/plaster/run/sigproc_v2/c_radiometry",
            c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        )
        lib = c.CDLL("./_radiometry.so")

    lib.context_init.argtypes = [
        c.POINTER(RadiometryContext),
    ]
    lib.context_init.restype = c.c_char_p

    lib.context_free.argtypes = [
        c.POINTER(RadiometryContext),
    ]

    lib.radiometry_field_stack_one_peak.argtypes = [
        c.POINTER(RadiometryContext),  # RadiometryContext context
        c_common_tools.typedef_to_ctype("Index"),  # Index peak_start_i,
        c_common_tools.typedef_to_ctype("Index"),  # Index n_peaks,
    ]
    lib.radiometry_field_stack_one_peak.restype = c.c_char_p

    _lib = lib
    return lib


@contextmanager
def context(chcy_ims, locs, reg_psf: RegPSF, focus_adjustment):
    """
    with radiometry.context(...) as ctx:
        zap.work_orders(do_radiometry, ...)

        ("n_channels", "Size"),
        ("n_cycles", "Size"),
        ("n_peaks", "Size"),
        ("n_divs", "Size"),
        ("peak_mea", "Size"),

        # Outputs
        ("out_radiometry", F64Arr),  # signal, noise, snr, aspect_ratio


    """
    lib = load_lib()

    check.array_t(chcy_ims, ndim=4, dtype=np.float64)
    n_channels, n_cycles, height, width = chcy_ims.shape

    check.array_t(locs, ndim=2, dtype=np.float64)
    check.affirm(locs.shape[1] == 2)
    n_peaks = locs.shape[0]

    check.t(reg_psf, RegPSF)
    peak_mea = reg_psf.peak_mea
    n_divs = reg_psf.n_divs

    check.array_t(focus_adjustment, ndim=1, dtype=np.float64)

    out_radiometry = np.zeros((n_peaks, 4), dtype=np.float64)
    ctx = RadiometryContext(
        chcy_ims=F64Arr.from_ndarray(chcy_ims),
        locs=F64Arr.from_ndarray(locs),
        reg_psf_params=F64Arr.from_ndarray(reg_psf.params),
        focus_adjustment=F64Arr.from_ndarray(focus_adjustment),
        n_channels=n_channels,
        n_cycles=n_cycles,
        n_peaks=n_peaks,
        n_divs=n_divs,
        peak_mea=peak_mea,
        height=height,
        width=width,
        out_radiometry=out_radiometry,
        _out_radiometry=out_radiometry,
    )

    error = lib.context_init(ctx)
    if error is not None:
        raise CException(error)

    try:
        yield ctx
    finally:
        lib.context_free(ctx)


def _do_radiometry_field_stack_one_peak(ctx: RadiometryContext, peak_i: int):
    """
    Worker for radiometry_field_stack() zap
    """
    lib = load_lib()
    error = lib.radiometry_field_stack_one_peak(ctx, peak_i)
    if error is not None:
        raise CException(error)


def radiometry_field_stack(chcy_ims, locs, reg_psf: RegPSF, focus_adjustment):
    with context(
        chcy_ims=chcy_ims,
        locs=locs,
        reg_psf=reg_psf,
        focus_adjustment=focus_adjustment,
    ) as ctx:
        check.array_t(locs, ndim=2, dtype=np.float64)
        n_peaks = locs.shape[0]
        zap.arrays(
            _do_radiometry_field_stack_one_peak,
            dict(cy_i=np.arange(n_peaks)),
            _process_mode=False,
            _trap_exceptions=False,
            ctx=ctx,
        )

    return ctx._out_radiometry
