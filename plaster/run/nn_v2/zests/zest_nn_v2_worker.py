import pandas as pd
import numpy as np
from munch import Munch
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_v2
from plaster.run.prep.prep_worker import prep
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep import prep_fixtures
from plaster.run.error_model import ErrorModel, GainModel
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.run.sigproc_v2.sigproc_v2_fixtures import simple_sigproc_v2_result_fixture
from zest import zest
from plaster.tools.log.log import debug


def zest_nn_v2_worker():
    prep_result = prep_fixtures.result_random_fixture(2)

    from plaster.run.nn_v2.c.nn_v2 import init as nn_v2_init

    nn_v2_init()

    def _run(labels="DE", sigproc_result=None, _prep_result=prep_result):
        sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(
            _prep_result, labels=labels
        )

        # Flip just to convince myself that it is working
        # (ie they aren't accidentally in the right order)
        sim_v2_result.test_radmat = np.flip(sim_v2_result.test_radmat, axis=0).copy()
        sim_v2_result.test_true_pep_iz = np.flip(
            sim_v2_result.test_true_pep_iz, axis=0
        ).copy()

        sim_v2_result.test_true_dye_iz = np.flip(
            sim_v2_result.test_true_dye_iz, axis=0
        ).copy()

        gain_model = sim_v2_result.params.error_model.to_gain_model()
        nn_v2_params = NNV2Params(n_neighbors=10, gain_model=gain_model,)

        nn_v2_result = nn_v2(
            nn_v2_params, _prep_result, sim_v2_result, sigproc_result=sigproc_result
        )

        return nn_v2_result, sim_v2_result

    def it_runs_single_channel():
        for tries in range(10):
            nn_v2_result, sim_v2_result = _run(labels="DE")
            trues = sim_v2_result.test_true_pep_iz
            n_right = (nn_v2_result.calls().pep_i == trues).sum()
            n_total = trues.shape[0]
            if n_right >= int(0.5 * n_total):
                break
        else:
            raise AssertionError("never exceeded 50%")

    def it_runs_multi_channel():
        prep_result = prep_fixtures.result_random_fixture(10)
        nn_v2_result, sim_v2_result = _run(labels="DE,ABC", _prep_result=prep_result)
        trues = sim_v2_result.test_true_pep_iz
        n_right = (nn_v2_result.calls().pep_i == trues).sum()
        n_total = trues.shape[0]
        assert n_right >= int(0.3 * n_total)

    @zest.skip(reason="WIP")
    def run_without_sigproc():
        nn_v2_result, sim_v2_result = _run(sigproc_result=None)

        a = (
            sim_v2_result.test_true_dye_iz == nn_v2_result._test_calls.dyt_i.values
        ).sum()
        debug(a)

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

    @zest.skip(reason="WIP")
    def it_runs_with_sigproc():
        raise NotImplementedError
        # TODO Need to deal with sigproc v2 calibration fixtures

        sigproc_result = simple_sigproc_v2_result_fixture(prep_result)
        nn_v2_result, sim_v2_result = _run(labels="DE", sigproc_result=sigproc_result)

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
            row_k_sigma=0.15,
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
        gain_model=ErrorModel(**sim_v2_params.error_model).to_gain_model()
    )
    nn_result = nn_v2(nn_v2_params, prep_result, sim_v2_result, None)
    df = nn_result.calls()
    assert np.all(df.pep_i == 1)
