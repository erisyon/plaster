import sys
import time
cimport lmfit
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from plaster.tools.schema import check
from cpython.exc cimport PyErr_CheckSignals


# Local helpers
def _assert_array_contiguous(arr, dtype, which):
    if not (isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]):
        raise AssertionError(
            f"array {which} is incorrect: "
            f"is_ndarray: {isinstance(arr, np.ndarray)} "
            f"dtype was {arr.dtype} expected {dtype} "
            f"continguous was {arr.flags['C_CONTIGUOUS']}."
        )


def _assert_with_trace(condition, message):
    """
    Cython assert doesn't produce useful traces
    """
    if not condition:
        raise AssertionError(message)


global_progress_callback = None

cdef void _progress(int complete, int total, int retry):
    # print(f"progress {complete} {total} {retry}", file=sys.stderr)
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


cdef int _check_keyboard_interrupt_callback():
    try:
        PyErr_CheckSignals()
    except KeyboardInterrupt:
        return 1
    return 0


# TODO: This needs to be refactored to allow multi thread

def fast_gauss_2d(peak_ims, guess_params, n_threads, progress=None):
    """
    This is the interface to the C implementation of Gaussian 2D fitter

    Arguments:
        peak_ims: ndarray((n_peaks, mea, mea), dtype=np.float64)
        guess_params: ndarray(7)
            The starting point for the fitter

    Returns:
        fits: ndarray(n_peaks, 7)
            amplitude, sigma_x, sigma_y, pos_x, pos_y, rho, offset
    """
    cdef c.PixType [:, :, ::1] peak_ims_view
    cdef double [::1] guess_params_view

    # CHECKS
    assert c.sanity_check() == 0
    _assert_array_contiguous(peak_ims, np.float64, "peak_ims")
    check.array_t(peak_ims, ndim=3)
    n_peaks, mea0, mea1 = peak_ims.shape
    _assert_with_trace(mea0 == mea1, "peak ims must be square")

    global global_progress_callback
    global_progress_callback = progress

    # ALLOCATE output arrays
    output_fits = np.zeros((n_peaks, 7), dtype=np.float64)

    # CREATE cython views
    peak_ims_view = peak_ims
    output_fits_view = output_fits
    guess_params_view = guess_params

    cdef c.Index peak_i
    cdef int mea = mea0
    cdef double params[7]

    for peak_i in range(n_peaks):
        PyErr_CheckSignals()
        memcpy(params, guess_params_view, sizeof(double)*7)
        lmfit.fit_gauss_2d(
            &peak_ims_view[peak_i, 0, 0],
            mea,
            params,
            None,  # double *info,
            None,  # double *covar
        )
        memcpy(&output_fits[peak_i, 0], params, sizeof(double)*7)

    return output_fits
