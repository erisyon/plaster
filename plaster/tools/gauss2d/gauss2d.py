import ctypes
import numpy as np
from plumbum import local, FG


_lib_gauss_2d = None

# Set recompile to True to force recompile which is handy if debugging zbs.lmfits
# Also, remember to create a symlink in internal/overloads like:
# zbs.lmfits -> /Users/zack/git/zbs.lmfits/
recompile = False


def load_lib():
    global _lib_gauss_2d
    if _lib_gauss_2d is not None:
        return _lib_gauss_2d

    if recompile:
        with local.env(DST_FOLDER=local.cwd, LEVMAR_FOLDER="/zbs.lmfits/levmar-2.6"):
            with local.cwd("/erisyon/internal/overloads/zbs.lmfits"):
                local["./build.sh"] & FG

        lib = ctypes.CDLL("./liblmfits.so")
    else:
        lib = ctypes.CDLL("/zbs.lmfits/liblmfits.so")

    # fit_gauss_2d()
    lib.fit_gauss_2d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.fit_gauss_2d.restype = ctypes.c_int

    # fit_gauss_2d_on_float_image()
    lib.fit_gauss_2d_on_float_image.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.fit_gauss_2d_on_float_image.restpye = ctypes.c_int

    # fit_array_of_gauss_2d_on_float_image()
    lib.fit_array_of_gauss_2d_on_float_image.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
        ),  # np_float32 *im
        ctypes.c_int,  # np_int64 im_h
        ctypes.c_int,  # np_int64 im_w
        ctypes.c_int,  # np_int64 mea
        ctypes.c_int,  # np_int64 n_peaks
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
    lib.fit_array_of_gauss_2d_on_float_image.restpye = ctypes.c_int

    _lib_gauss_2d = lib
    return lib
