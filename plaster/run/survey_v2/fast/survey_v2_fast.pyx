import sys
import time
cimport c_survey_v2_fast as csurvey
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from plaster.tools.log.log import important


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]


global_progress_callback = None


cdef void _progress(int complete, int total, int retry):
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


# Wrapper for survey that prepares buffers for csurvey
def survey(
    dyemat,
    dyepeps,
    n_threads=1,
    rng_seed=None,
    progress=None,
):

