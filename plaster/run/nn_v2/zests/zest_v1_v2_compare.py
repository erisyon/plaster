from plaster.run.prep import prep_fixtures
from plaster.run.sim_v1 import sim_v1_fixtures
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.nn_v1.nn_v1_params import NNV1Params
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v1 import nn_v1_worker
from plaster.run.nn_v2 import nn_v2_worker
from zest import zest

def zest_v1_v2_compare():
    prep_result = prep_fixtures.result_random_fixture(5)

    sim_v1_result = sim_v1_fixtures.result_from_prep_fixture(prep_result)
    sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(prep_result)

    # TODO: Finish
    nn_v1_params = NNV1Params()
    nn_v1_result = nn_v1_worker.nn_v1(nn_v1_params, prep_result, sim_v1_result)

    nn_v2_params = NNV2Params()
    nn_v2_result = nn_v2_worker.nn_v2(nn_v2_params, sim_v2_result)

    zest()