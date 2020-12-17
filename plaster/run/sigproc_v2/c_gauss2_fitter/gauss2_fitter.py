import numpy as np
from plumbum import local
import ctypes as c
from enum import IntEnum
from plaster.tools.schema import check
from plaster.run.sigproc_v2.c_gauss2_fitter.build import build
from plaster.tools.c_common.c_common_tools import CException
from plaster.tools.log.log import debug


class Gauss2Params:
    AMP = 0
    SIGMA_X = 1
    SIGMA_Y = 2
    CENTER_X = 3
    CENTER_Y = 4
    RHO = 5
    OFFSET = 6
    N_PARAMS = 7  # Number above this point


class AugmentedGauss2Params(Gauss2Params):
    # These must match in gauss2_fitter.h
    MEA = 7
    NOISE = 8
    ASPECT_RATIO = 9
    N_FULL_PARAMS = 10


c_gauss_fitter_path = local.path("/erisyon/plaster/plaster/run/sigproc_v2/c_gauss2_fitter")


def init():
    """
    Must be called before anything else in this module
    """
    debug("BUILD C GAUSS FITTER")
    with local.cwd(c_gauss_fitter_path):
        build(
            dst_folder=c_gauss_fitter_path,
            c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        )
        lib = c.CDLL("./_gauss2_fitter.so")


_lib = None


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib = c.CDLL(c_gauss_fitter_path / "_gauss2_fitter.so")

    lib.gauss2_check.argtypes = []
    lib.gauss2_check.restype = c.c_char_p

    lib.gauss_2d.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        ),  # double *p
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        ),  # double *dst_x
        c.c_int,  # int m
        c.c_int,  # int n
        c.c_void_p,  # void *data
    ]

    lib.fit_array_of_gauss_2d_on_float_image.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"
        ),  # np_float64 *im
        c.c_int,  # np_int64 im_w
        c.c_int,  # np_int64 im_h
        c.c_int,  # np_int64 mea
        c.c_int,  # np_int64 n_peaks
        np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_int64 *center_x
        np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_int64 *center_y
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_float64 *params
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_float64 *var_params
        np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_int64 *fail
    ]
    lib.fit_array_of_gauss_2d_on_float_image.restype = c.c_char_p

    _lib = lib
    return lib


class Gauss2FitParams(IntEnum):
    # These must match in gauss2_fitter.h
    AMP = 0
    SIGNAL = 0  # Alias for AMP
    SIGMA_X = 1
    SIGMA_Y = 2
    CENTER_X = 3
    CENTER_Y = 4
    RHO = 5
    OFFSET = 6
    N_FIT_PARAMS = 7  # Number above this point
    MEA = 7
    NOISE = 8
    ASPECT_RATIO = 9
    N_FULL_PARAMS = 10


def gauss2(params):
    params = np.ascontiguousarray(params, dtype=np.float64)

    im = np.zeros((11, 11))
    im = np.ascontiguousarray(im.flatten(), dtype=np.float64)
    lib = load_lib()
    lib.gauss_2d(params, im, 7, 11 * 11, 0)

    return im.reshape((11, 11))


def fit_image(im, locs, guess_params, peak_mea):
    lib = load_lib()

    n_locs = int(len(locs))

    check.array_t(im, ndim=2, dtype=np.float64)
    im = np.ascontiguousarray(im, dtype=np.float64)
    # assert np.all(~np.isnan(im))

    check.array_t(im, ndim=2, dtype=np.float64, c_contiguous=True)
    check.array_t(locs, ndim=2, shape=(None, 2))

    locs_y = np.ascontiguousarray(locs[:, 0], dtype=np.int64)
    locs_x = np.ascontiguousarray(locs[:, 1], dtype=np.int64)
    locs_y[np.isnan(locs[:, 0])] = -1
    locs_x[np.isnan(locs[:, 1])] = -1

    fit_fails = np.zeros((n_locs,), dtype=np.int64)
    check.array_t(fit_fails, dtype=np.int64, c_contiguous=True)

    check.array_t(
        guess_params,
        dtype=np.float64,
        ndim=2,
        shape=(n_locs, AugmentedGauss2Params.N_FULL_PARAMS,),
    )

    fit_params = guess_params.copy()
    fit_params[:, AugmentedGauss2Params.MEA] = peak_mea
    fit_params = np.ascontiguousarray(fit_params.flatten())

    std_params = np.zeros((n_locs, AugmentedGauss2Params.N_FULL_PARAMS))
    std_params = np.ascontiguousarray(std_params.flatten())

    check.array_t(
        fit_params,
        dtype=np.float64,
        c_contiguous=True,
        ndim=1,
        shape=(n_locs * AugmentedGauss2Params.N_FULL_PARAMS,),
    )

    error = lib.gauss2_check()
    if error is not None:
        raise CException(error)

    error = lib.fit_array_of_gauss_2d_on_float_image(
        im,
        im.shape[1],  # Note inversion of axis (y is primary in numpy)
        im.shape[0],
        peak_mea,
        n_locs,
        locs_x,
        locs_y,
        fit_params,
        std_params,
        fit_fails,
    )
    if error is not None:
        raise CException(error)

    # RESHAPE fit_params and NAN-out any where the fit failed
    fit_params = fit_params.reshape((n_locs, AugmentedGauss2Params.N_FULL_PARAMS))
    fit_params[fit_fails == 1, :] = np.nan

    # After some very basic analysis, it seems that the following
    # parameters are reasonable guess for out of bound on the
    # std of fit.
    # Note, this analysis was done on 11x11 pixels and might
    # need to be different for other sizes.
    # BUT! after using this they seemed to knock out everything
    # so apparently the are not well tuned yet so this block is
    # temporarily removed.

    """
    std_params = std_params.reshape((n_locs, AugmentedGauss2Params.N_FULL_PARAMS))

    param_std_of_fit_limits = np.array((500, 0.18, 0.18, 0.15, 0.15, 0.08, 5,))

    out_of_bounds_mask = np.any(
        std_params[:, 0 : AugmentedGauss2Params.N_FIT_PARAMS]
        > param_std_of_fit_limits[None, :],
        axis=1,
    )

    fit_params[out_of_bounds_mask, :] = np.nan
    """

    return fit_params, std_params
