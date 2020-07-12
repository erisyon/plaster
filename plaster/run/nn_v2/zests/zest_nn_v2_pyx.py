from plaster.run.nn_v2.fast import nn_v2_fast
from plaster.tools.log.log import debug, prof
from zest import zest


def zest_nn_v2_pyx_runs():
    nn_v2_fast.test_nn()
