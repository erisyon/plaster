from plaster.run.nn_v2.fast import nn_v2_fast
from plaster.run.nn_v2.nn_v2_result import NNV2Result


def nn_worker(nn_v2_params, prep_result, sim_v2_result):
    nn_v2_fast.nn(nn_v2_params, prep_result, sim_v2_result)

    return NNV2Result(params=nn_v2_params,)
