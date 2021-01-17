import os
import numpy as np
from plumbum import local
import ctypes as c
from io import StringIO
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plaster.run.sigproc_v2.c_radiometry.build import build
from plaster.run.calib.calib import RegPSF
from plaster.tools.c_common.c_common_tools import CException
from plaster.tools.utils import utils
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import F64Arr
from plaster.tools.log.log import debug, important, prof


class RadiometryContext(c_common_tools.FixupStructure):
    # Operates on a single field stack so that we don't have
    # to realize all fields in to memory simultaneously

    # fmt: off
    _fixup_fields = [
        ("chcy_ims", F64Arr),  # Sub-pixel aligned  (n_channels, n_cycles, height, width)
        ("locs", F64Arr),  # Sub-pixel centered  (n_peaks, 2), where 2 is: (y, x)
        ("focus_adjustment", F64Arr),  # focus adjustment per cycle (n_cycles)
        ("reg_psf_samples", F64Arr),  # Reg_psf samples (n_divs, n_divs, 3) (y, x, (sig_x, sig_y, rho))

        # Parameters
        ("n_channels", "Size"),
        ("n_cycles", "Size"),
        ("n_peaks", "Size"),
        ("n_divs", "Float64"),
        ("peak_mea", "Size"),
        ("height", "Float64"),
        ("width", "Float64"),
        ("raw_height", "Float64"),
        ("raw_width", "Float64"),

        # Outputs
        ("out_radiometry", F64Arr),  # (n_peaks, n_channels, n_cycles, 4), where 4 is: (signal, noise, snr, aspect_ratio)
    ]
    # fmt: on


c_radiometry_path = local.path("/erisyon/plaster/plaster/run/sigproc_v2/c_radiometry")


def init():
    """
    This must be called once before any work
    """
    RadiometryContext.struct_fixup()

    with local.cwd(c_radiometry_path):
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


_lib = None


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib = c.CDLL(c_radiometry_path / "_radiometry.so")
    lib.context_init.argtypes = [
        c.POINTER(RadiometryContext),
    ]
    lib.context_init.restype = c.c_char_p

    lib.context_free.argtypes = [
        c.POINTER(RadiometryContext),
    ]

    lib.radiometry_field_stack_one_peak.argtypes = [
        c.POINTER(RadiometryContext),  # RadiometryContext context
        c_common_tools.typedef_to_ctype("Index"),  # Index peak_i,
    ]
    lib.radiometry_field_stack_one_peak.restype = c.c_char_p

    lib.test_interp.argtypes = [
        c.POINTER(RadiometryContext),  # RadiometryContext context
        c_common_tools.typedef_to_ctype("Float64"),  # Float64 loc_x
        c_common_tools.typedef_to_ctype("Float64"),  # Float64 loc_y
        c.c_void_p,  # Float64 *out_vals
    ]
    lib.test_interp.restype = c.c_char_p

    _lib = lib
    return lib


@contextmanager
def context(chcy_ims, locs, reg_psf: RegPSF, focus_adjustment):
    """
    with radiometry.context(...) as ctx:
        zap.work_orders(do_radiometry, ...)
    """
    lib = load_lib()

    check.array_t(chcy_ims, ndim=4, dtype=np.float64)
    n_channels, n_cycles, height, width = chcy_ims.shape

    check.array_t(locs, ndim=2, dtype=np.float64)
    check.affirm(locs.shape[1] == 2)
    n_peaks = locs.shape[0]

    check.t(reg_psf, RegPSF)
    peak_mea = reg_psf.peak_mea

    check.array_t(focus_adjustment, ndim=1, dtype=np.float64)

    # I previously attempted to use a spline interpolator inside the
    # radiometry.c but had problems so I reverted (at least for now)
    # to a high-res sampling of the reg_psf
    n_divs = 64
    assert n_channels == 1  # Until multi-channel
    samples = reg_psf.sample_params_grid(ch_i=0, n_divs=n_divs)

    out_radiometry = np.zeros((n_peaks, n_channels, n_cycles, 4), dtype=np.float64)

    reg_psf_samples = np.ascontiguousarray(samples)

    ctx = RadiometryContext(
        chcy_ims=F64Arr.from_ndarray(chcy_ims),
        locs=F64Arr.from_ndarray(locs),
        _locs=locs,
        focus_adjustment=F64Arr.from_ndarray(focus_adjustment),
        n_channels=n_channels,
        n_cycles=n_cycles,
        n_peaks=n_peaks,
        n_divs=n_divs,
        peak_mea=peak_mea,
        height=height,
        width=width,
        raw_height=reg_psf.im_mea,
        raw_width=reg_psf.im_mea,
        n_reg_psf_samples=len(samples),
        reg_psf_samples=F64Arr.from_ndarray(reg_psf_samples),
        out_radiometry=F64Arr.from_ndarray(out_radiometry),
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
        if n_peaks > 0:
            with zap.Context(trap_exceptions=False, mode="thread"):
                zap.arrays(
                    _do_radiometry_field_stack_one_peak,
                    dict(peak_i=np.arange(n_peaks)),
                    ctx=ctx,
                )

    # Sanity check
    bad_signals = ctx._out_radiometry[:, :, :, 0] > 1e6
    if np.any(bad_signals):
        important(
            f"there were {bad_signals.sum()} of {len(bad_signals)} bad radiometries. Converting to nan"
        )
        ctx._out_radiometry[bad_signals, :] = np.nan

    return ctx._out_radiometry


def test_interp():
    reg_psf = RegPSF.fixture_variable()
    with context(
        chcy_ims=np.ones((1, 1, 512, 512)),
        locs=np.zeros((1, 2)),
        reg_psf=reg_psf,
        focus_adjustment=np.ones((1,)),
    ) as ctx:
        lib = load_lib()

        diffs = []

        for y in np.linspace(0, 511, 17):
            for x in np.linspace(0, 511, 17):
                result = np.zeros((3,))
                error = lib.test_interp(ctx, x, y, result.ctypes.data_as(c.c_void_p))
                if error is not None:
                    raise CException(error)

                sig_x = reg_psf.interp_sig_x_fn[0](x, y)
                sig_y = reg_psf.interp_sig_y_fn[0](x, y)
                rho = reg_psf.interp_rho_fn[0](x, y)

                diffs += [
                    (
                        x,
                        y,
                        sig_x[0],
                        sig_y[0],
                        rho[0],
                        sig_x[0] - result[0],
                        sig_y[0] - result[1],
                        rho[0] - result[2],
                    )
                ]

        diffs = np.array(diffs)
        diffs = np.abs(diffs[:, 5:])
        return np.all(diffs < 0.03)

    return ctx._out_radiometry
