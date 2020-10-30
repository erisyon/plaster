import pandas as pd
import numpy as np
from munch import Munch
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_v2
from zest import zest
from plaster.tools.log.log import debug


def zest_nn_v2_result():
    def it_returns_calls():
        raise NotImplementedError

    def it_returns_all():
        raise NotImplementedError

    def it_filters_nul_calls():
        raise NotImplementedError

    def it_filters_k_range():
        raise NotImplementedError

    def it_filters_k_score():
        raise NotImplementedError

    zest()
