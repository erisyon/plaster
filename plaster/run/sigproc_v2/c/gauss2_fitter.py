import ctypes as c
import numpy as np
from plumbum import local, FG
import ctypes as c
from contextlib import contextmanager
from plaster.tools.schema import check
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import Tab
from plaster.run.sigproc_v2.c.build import build
from plaster.tools.log.log import debug


_lib = None


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    with local.cwd("/erisyon/plaster/plaster/run/sigproc_v2/c"):
        build(
            dst_folder="/erisyon/plaster/plaster/run/nn_v2/c",
            c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        )
        lib = c.CDLL("./_gauss2_fitter.so")

    lib.sanity_check()

    lib.fit_array_of_gauss_2d_on_float_image.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
        ),  # np_float32 *im
        c.c_int,  # np_int64 im_h
        c.c_int,  # np_int64 im_w
        c.c_int,  # np_int64 mea
        c.c_int,  # np_int64 n_peaks
        np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_int64 *center_y
        np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_int64 *center_x
        np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_float64 *params
        np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"
        ),  # np_int64 *fail
    ]
    lib.fit_array_of_gauss_2d_on_float_image.restpye = c.c_int

    _lib = lib
    return lib


def do_fit_image(im, locs, psf_params):
    lib = load_lib()
    raise NotImplementedError
    lib.fit_array_of_gauss_2d_on_float_image()
