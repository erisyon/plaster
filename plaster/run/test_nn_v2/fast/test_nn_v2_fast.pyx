import time
cimport ctest_nn_v2_fast as ctest_nn
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free


# Local helpers
def _assert_array_contiguous(arr, dtype):
    assert isinstance(arr, np.ndarray) and arr.dtype == dtype and arr.flags["C_CONTIGUOUS"]


def test_nn():
        ctest_nn.test_flann()
