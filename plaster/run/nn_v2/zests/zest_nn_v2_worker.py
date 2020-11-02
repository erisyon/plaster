import pandas as pd
import numpy as np
from munch import Munch
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_v2
from plaster.run.prep.prep_worker import prep
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.run.sigproc_v2.sigproc_v2_fixtures import simple_sigproc_v2_result_fixture
from zest import zest
from plaster.tools.log.log import debug


def zest_nn_v2_worker():
    prep_result = prep_fixtures.result_random_fixture(2)

    sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(prep_result, labels="DE")

    # Flip just to convince myself that it is working
    # (ie they aren't accidentally in the right order)
    sim_v2_result.test_radmat = np.flip(sim_v2_result.test_radmat, axis=0).copy()
    sim_v2_result.test_true_pep_iz = np.flip(
        sim_v2_result.test_true_pep_iz, axis=0
    ).copy()

    nn_v2_params = NNV2Params(
        n_neighbors=4,
        beta=5000.0,
        sigma=0.20,
        zero_beta=0.0,
        zero_sigma=200.0,
        row_k_std=0.0,
    )

    def it_runs_without_sigproc():
        nn_v2_result = nn_v2(
            nn_v2_params, prep_result, sim_v2_result, sigproc_result=None
        )

    @zest.skip(reason="Need to deal with sigproc v2 calibration fixtures")
    def it_runs_with_sigproc():
        sigproc_result = simple_sigproc_v2_result_fixture(prep_result)
        nn_v2_result = nn_v2(
            nn_v2_params, prep_result, sim_v2_result, sigproc_result=sigproc_result
        )

    zest()


def zest_v2_stress_like_e2e():
    # This was dying with a "double free or corruption (!prev)"
    # This was a bug in n_dyetracks counting now fixed, but leaving this test in for regression.

    prep_params = PrepParams(
        decoy_mode=None,
        n_peps_limit=None,
        n_ptms_limit=5,
        protease=None,
        proteins=[
            Munch(
                abundance=None,
                name="pep25",
                ptm_locs="",
                report=0,
                sequence="GCAGCAGAG ",
            )
        ],
        proteins_of_interest=[],
    )
    pro_spec_df = pd.DataFrame(prep_params.proteins)
    prep_result = prep(prep_params, pro_spec_df)

    sim_v2_param_block = Munch(
        allow_train_test_to_be_identical=False,
        dyes=[Munch(channel_name="ch_0", dye_name="dye_0")],
        enable_ptm_labels=False,
        error_model=Munch(
            dyes=[
                Munch(
                    dye_name="dye_0",
                    p_bleach_per_cycle=0.05,
                    p_non_fluorescent=0.07,
                    beta=7500.0,
                    sigma=0.16,
                    zero_beta=300.0,
                    zero_sigma=700.0,
                )
            ],
            labels=[
                Munch(
                    label_name="label_0",
                    p_failure_to_attach_to_dye=0.0,
                    p_failure_to_bind_amino_acid=0.0,
                )
            ],
            p_detach=0.05,
            p_edman_failure=0.06,
            row_k_sigma=0.0,
        ),
        is_survey=False,
        labels=[
            Munch(
                amino_acid="C", dye_name="dye_0", label_name="label_0", ptm_only=False,
            )
        ],
        n_edmans=8,
        n_mocks=1,
        n_pres=0,
        n_samples_test=1000,
        n_samples_train=5000,
        random_seed=None,
        test_includes_dyemat=False,
        train_includes_radmat=False,
    )

    sim_v2_params = SimV2Params(include_dfs=True, **sim_v2_param_block)

    sim_v2_result = sim_v2_worker.sim_v2(sim_v2_params, prep_result)
    sim_v2_result._generate_flu_info(prep_result)

    nn_v2_params = NNV2Params(
        beta=5000.0, sigma=0.20, zero_beta=0.0, zero_sigma=200.0, row_k_std=0.0,
    )
    nn_result = nn_v2(nn_v2_params, prep_result, sim_v2_result, None)
    assert np.all(nn_result.test_pred_pep_iz == 1)
