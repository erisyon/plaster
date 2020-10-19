import pandas as pd
from munch import Munch
import numpy as np
from plaster.tools.utils.tmp import tmp_folder, tmp_file
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v1 import sim_v1_fixtures
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2 import nn_v2_worker
from zest import zest
from plaster.tools.log.log import debug, prof
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_worker import prep


@zest.skip(reason="V1 deprecated")
def zest_v1_v2_compare():
    with tmp_folder(chdir=True):
        prep_result = prep_fixtures.result_random_fixture(2)

        sim_v1_result = sim_v1_fixtures.result_from_prep_fixture(
            prep_result, labels="A,B"
        )
        sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(
            prep_result, labels="A,B"
        )

        nn_v1_params = NNV1Params()
        nn_v1_result = nn_v1_worker.nn_v1(nn_v1_params, prep_result, sim_v1_result)

        nn_v2_params = NNV2Params()
        nn_v2_result = nn_v2_worker.nn_v2(
            nn_v2_params, prep_result, sim_v2_result, None
        )

        n_correct_v1 = np.sum(
            sim_v1_result.test_true_pep_iz == nn_v1_result.test_pred_pep_iz
        )
        n_correct_v2 = np.sum(
            sim_v2_result.test_true_pep_iz == nn_v2_result.test_pred_pep_iz
        )

        assert (n_correct_v1 - n_correct_v2) ** 2 < 200 ** 2
        assert (
            sim_v1_result.test_true_pep_iz.shape == sim_v2_result.test_true_pep_iz.shape
        )

    zest()


def zest_v2_stress_like_e2e():
    # This was dieing with a "double free or corruption (!prev)"
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
                    beta=7500.0,
                    dye_name="dye_0",
                    gain=7500.0,
                    p_bleach_per_cycle=0.05,
                    p_non_fluorescent=0.07,
                    sigma=0.16,
                    vpd=0.1,
                    zero_mean=0.0,
                    zero_sigma=200.0,
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

    nn_v2_params = NNV2Params()
    nn_v2_worker.nn_v2(nn_v2_params, prep_result, sim_v2_result, None)


# def zest_v2_count():
#     with tmp_folder(chdir=True):
#         prep_result = prep_fixtures.result_random_fixture(5000)
#
#         sim_v2_fixtures.result_from_prep_fixture(prep_result, n_labels=3, n_edmans=15)
#     zest()
