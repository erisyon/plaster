import numpy as np

from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_v2
from plaster.run.prep.prep_result import PrepResult
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel
from zest import zest


def zest_nn_v2_worker():
    prep_result = prep_fixtures.result_simple_fixture()

    error_model = ErrorModel.no_errors(n_channels=2, beta=100.0)

    sim_v2_params = SimV2Params.construct_from_aa_list(
        ["A", "B"],
        error_model=error_model,
        n_edmans=4,
        n_samples_train=10,
        n_samples_test=5,
    )

    sim_v2_result = sim_v2_worker.sim_v2(sim_v2_params, prep_result)

    # Flip just to convinvce myself that it is working
    # so that they aren't accidentally in the right order
    sim_v2_result.test_radmat = np.flip(sim_v2_result.test_radmat, axis=0).copy()
    sim_v2_result.test_true_pep_iz = np.flip(sim_v2_result.test_true_pep_iz, axis=0).copy()

    nn_v2_params = NNV2Params(n_neighbors=2)

    nn_v2_result = nn_v2(nn_v2_params, sim_v2_result)

    def it_predicts_test():
        assert nn_v2_result.test_pred_pep_iz.tolist() == [3] * 5 + [2] * 5 + [1] * 5
        assert np.all(nn_v2_result.test_scores >= 1.0)

    zest()
