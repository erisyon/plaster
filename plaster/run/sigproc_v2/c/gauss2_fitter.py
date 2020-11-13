import ctypes as c
import numpy as np
from plumbum import local, FG
import ctypes as c
from contextlib import contextmanager
from plaster.tools.schema import check
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import Tab
from plaster.run.sigproc_v2.c.build import build
from plaster.tools.image import imops, coord
from plaster.tools.log.log import debug


_lib = None


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    with local.cwd("/erisyon/plaster/plaster/run/sigproc_v2/c"):
        build(
            dst_folder="/erisyon/plaster/plaster/run/sigproc_v2/c",
            c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        )
        lib = c.CDLL("./_gauss2_fitter.so")

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
            dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
        ),  # np_float32 *im
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


# TODO: Mo ve this and NNV2Exception to a generic CException
class Gauss2FitException(Exception):
    def __init__(self, s):
        super().__init__(s.decode("ascii"))


class Gauss2FitParams:
    # These must match in gauss2_fitter.h
    SIGNAL = 0
    NOISE = 1
    ASPECT_RATIO = 2
    FIRST_FIT_PARAM = 3
    AMP = 3
    SIGMA_X = 4
    SIGMA_Y = 5
    CENTER_X = 6
    CENTER_Y = 7
    RHO = 8
    OFFSET = 9
    LAST_FIT_PARAM = 10
    MEA = 10
    N_FULL_PARAMS = 11
    N_FIT_PARAMS = 7


def gauss2(params):
    params = np.ascontiguousarray(params, dtype=np.float64)

    im = np.zeros((11, 11))
    im = np.ascontiguousarray(im.flatten(), dtype=np.float64)
    lib = load_lib()
    lib.gauss_2d(params, im, 7, 11 * 11, 0)

    return im.reshape((11, 11))


def fit_image(im, locs, fit_params, psf_mea):
    lib = load_lib()

    n_locs = int(len(locs))

    im = np.ascontiguousarray(im, dtype=np.float32)

    check.array_t(im, ndim=2, dtype=np.float32, c_contiguous=True)
    check.array_t(locs, ndim=2, shape=(None, 2))

    locs_y = np.ascontiguousarray(locs[:, 0], dtype=np.int64)
    locs_x = np.ascontiguousarray(locs[:, 1], dtype=np.int64)

    fit_fails = np.zeros((n_locs,), dtype=np.int64)
    check.array_t(fit_fails, dtype=np.int64, c_contiguous=True)

    check.array_t(
        fit_params,
        dtype=np.float64,
        ndim=2,
        shape=(n_locs, Gauss2FitParams.N_FIT_PARAMS,),
    )

    std_params = np.zeros((n_locs, Gauss2FitParams.N_FIT_PARAMS))
    std_params = np.ascontiguousarray(std_params.flatten())

    ret_params = np.zeros((n_locs, Gauss2FitParams.N_FULL_PARAMS))
    ret_params[
        :, Gauss2FitParams.FIRST_FIT_PARAM : Gauss2FitParams.LAST_FIT_PARAM
    ] = fit_params
    ret_params[:, Gauss2FitParams.MEA] = psf_mea
    ret_params = np.ascontiguousarray(ret_params.flatten())

    check.array_t(
        ret_params,
        dtype=np.float64,
        c_contiguous=True,
        ndim=1,
        shape=(n_locs * Gauss2FitParams.N_FULL_PARAMS,),
    )

    error = lib.gauss2_check()
    if error is not None:
        raise Gauss2FitException(error)

    # debug(locs_y[0], locs_x[0])
    # _im = imops.crop(im, off=coord.XY(locs_x[0], locs_y[0]), dim=coord.WH(11, 11), center=True)
    # np.save("_im.npy", _im)

    error = lib.fit_array_of_gauss_2d_on_float_image(
        im,
        im.shape[1],  # Note inversion of axis (y is primary in numpy)
        im.shape[0],
        psf_mea,
        n_locs,
        locs_x,
        locs_y,
        ret_params,
        std_params,
        fit_fails,
    )
    if error is not None:
        raise Gauss2FitException(error)

    # debug(n_locs)
    # n_fails = (fit_fails == 1).sum()
    # debug(n_fails)

    # After some very basic analysis, it seems that

    ret_params = ret_params.reshape((n_locs, Gauss2FitParams.N_FULL_PARAMS))
    ret_params[fit_fails == 1, :] = np.nan
    return ret_params, std_params
