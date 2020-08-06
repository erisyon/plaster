import numpy as np

from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_v2
from plaster.run.prep.prep_result import PrepResult
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel
from plaster.run.sigproc_v1.sigproc_v1_fixtures import simple_sigproc_result_fixture
from zest import zest


def zest_nn_v2_worker():
    prep_result = prep_fixtures.result_random_fixture(2)

    sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(prep_result)

    # Flip just to convince myself that it is working
    # (ie they aren't accidentally in the right order)
    sim_v2_result.test_radmat = np.flip(sim_v2_result.test_radmat, axis=0).copy()
    sim_v2_result.test_true_pep_iz = np.flip(
        sim_v2_result.test_true_pep_iz, axis=0
    ).copy()

    nn_v2_params = NNV2Params(n_neighbors=4)

    def it_predicts_test():
        nn_v2_result = nn_v2(
            nn_v2_params, prep_result, sim_v2_result, sigproc_result=None
        )
        assert nn_v2_result.shape == (3000,)

    def it_classifies_sigproc():
        sigproc_result = simple_sigproc_result_fixture()
        nn_v2_result = nn_v2(
            nn_v2_params, prep_result, sim_v2_result, sigproc_result=sigproc_result
        )

    zest()
