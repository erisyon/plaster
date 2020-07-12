import time
cimport ctest_nn_v2_fast as ctest_nn
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free

def test_nn():
        ctest_nn.test_flann()
