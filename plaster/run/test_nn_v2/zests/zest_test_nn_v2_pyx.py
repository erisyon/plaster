from plaster.run.test_nn_v2.fast import test_nn_v2_fast
from plaster.tools.log.log import debug, prof
from zest import zest


def zest_test_nn_pyx_runs():
    test_nn_v2_fast.test_nn()
