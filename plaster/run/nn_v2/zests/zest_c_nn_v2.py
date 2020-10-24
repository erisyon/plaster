import numpy as np
from plaster.run.nn_v2.c import nn_v2 as c_nn_v2

from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_v2
from plaster.run.prep.prep_result import PrepResult
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel

# from plaster.run.sigproc_v1.sigproc_v1_fixtures import simple_sigproc_result_fixture
from zest import zest
from plaster.tools.log.log import debug


def zest_c_nn_v2():
    dyemat = np.array([[0, 0, 0], [3, 2, 1], [2, 1, 1], [1, 0, 0],])

    dyepeps = np.array(
        [
            # (dyt_i, pep_i, count)
            [1, 1, 10],
            [2, 1, 10],
            [3, 1, 10],
            [1, 2, 30],
        ]
    )

    # TODO: radmat from dyemat given model
    here

    nn_v2_context = c_nn_v2.context_create(
        dyemat,
        dyepeps,
        test_radmat,
        nn_v2_params.beta,
        nn_v2_params.sigma,
        nn_v2_params.zero_beta,
        nn_v2_params.zero_sigma,
        nn_v2_params.row_k_std,
        n_neighbors=nn_v2_params.n_neighbors,
        run_row_k_fit=nn_v2_params.run_row_k_fit,
        run_against_all_dyetracks=nn_v2_params.run_against_all_dyetracks,
        progress=progress,
    )
