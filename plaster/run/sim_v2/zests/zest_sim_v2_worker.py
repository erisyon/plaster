from munch import Munch
import pandas as pd
import numpy as np
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel, RadType
from plaster.run.sim_v2 import sim_v2_worker
from plaster.tools.utils import utils
from zest import zest
from plaster.tools.log.log import debug, prof


def zest_sample_pep_dyemat():
    def it_samples():
        dyepep_group = np.array([[1, 1, 5], [2, 1, 5], [3, 1, 0],], dtype=int)

        n_samples_per_pep = 10
        sampled_dt_iz = sim_v2_worker._sample_pep_dyemat(
            dyepep_group, n_samples_per_pep
        )
        assert sampled_dt_iz.shape == (n_samples_per_pep,)
        assert not np.any(sampled_dt_iz == 0)
        assert np.any(sampled_dt_iz == 1)
        assert np.any(sampled_dt_iz == 2)
        assert not np.any(sampled_dt_iz == 3)

    def it_handles_no_samples():
        dyepep_group = np.zeros((0, 3), dtype=int)
        n_samples_per_pep = 10
        sampled_dt_iz = sim_v2_worker._sample_pep_dyemat(
            dyepep_group, n_samples_per_pep
        )
        assert sampled_dt_iz.shape == (0,)

    def it_handles_no_counts():
        dyepep_group = np.zeros((1, 3), dtype=int)
        n_samples_per_pep = 10
        sampled_dt_iz = sim_v2_worker._sample_pep_dyemat(
            dyepep_group, n_samples_per_pep
        )
        assert sampled_dt_iz.shape == (0,)

    zest()


def zest_radmat_from_sampled_pep_dyemat():
    n_channels = 2
    n_cycles = 3
    n_samples_per_pep = 5
    n_peps = 2

    # fmt: off
    sampled_dyemat = np.array([
        [[1, 1, 0], [1, 0, 0]],
        [[2, 2, 1], [2, 1, 0]],
        [[2, 2, 1], [2, 1, 0]],
        [[1, 1, 0], [1, 0, 0]],
        [[1, 1, 0], [1, 0, 0]],
    ], dtype=np.uint8)
    # fmt: on

    ch_params_no_noise = [
        Munch(beta=10.0, sigma=0.0, zero_beta=0.0, zero_sigma=0.0),
        Munch(beta=10.0, sigma=0.0, zero_beta=0.0, zero_sigma=0.0),
    ]

    ch_params_with_noise = [
        Munch(beta=10.0, sigma=0.1, zero_beta=0.0, zero_sigma=200.0),
        Munch(beta=10.0, sigma=0.1, zero_beta=0.0, zero_sigma=200.0),
    ]

    output_radmat = None

    def _before():
        nonlocal output_radmat
        output_radmat = np.zeros(
            (n_peps, n_samples_per_pep, n_channels, n_cycles), dtype=np.float32
        )

    def it_returns_noise_free_radmat():
        sim_v2_worker._radmat_from_sampled_pep_dyemat(
            sampled_dyemat, ch_params_no_noise, n_channels, output_radmat, pep_i=1
        )

        assert output_radmat.shape == (n_peps, n_samples_per_pep, n_channels, n_cycles)

        # Peptide 0 is all zero:
        assert np.all(output_radmat[0, :, :, :] == 0.0)

        # Peptide 1 is noise-free
        assert np.all(
            output_radmat[1, :, :, :] == 10.0 * sampled_dyemat.astype(np.float32)
        )

    def it_returns_noisy_radmat():
        sim_v2_worker._radmat_from_sampled_pep_dyemat(
            sampled_dyemat, ch_params_with_noise, n_channels, output_radmat, pep_i=1
        )

        assert output_radmat.shape == (n_peps, n_samples_per_pep, n_channels, n_cycles)
        assert np.all(output_radmat[0, :, :, :] == 0.0)
        expected = 10.0 * sampled_dyemat.astype(np.float32)
        diff = output_radmat[1, :, :, :] - expected
        diff = utils.np_safe_divide(diff, expected) ** 2
        if not (np.all((diff ** 2 < 0.15 ** 2) | np.isnan(diff))):
            debug(diff)
        assert np.all((diff ** 2 < 0.15 ** 2) | np.isnan(diff))

    def it_handles_empty_dyemat():
        empty_dyemat = np.zeros((0, n_channels, n_cycles), dtype=np.uint8)

        sim_v2_worker._radmat_from_sampled_pep_dyemat(
            empty_dyemat, ch_params_no_noise, n_channels, output_radmat, pep_i=1
        )

        assert output_radmat.shape == (n_peps, n_samples_per_pep, n_channels, n_cycles)
        assert np.all(output_radmat[:, :, :, :] == 0.0)

    zest()


def zest_radmat_sim():
    ch_params_with_noise = [
        Munch(beta=7500.0, sigma=0.16, zero_beta=0.0, zero_sigma=200),
        Munch(beta=7500.0, sigma=0.16, zero_beta=0.0, zero_sigma=200),
    ]

    ch_params_no_noise = [
        Munch(beta=1.0, sigma=0.0, zero_beta=0.0, zero_sigma=0.0),
        Munch(beta=1.0, sigma=0.0, zero_beta=0.0, zero_sigma=0.0),
    ]

    # fmt: off
    dyemat = np.array([
        [[0, 0, 0], [0, 0, 0]],
        [[1, 1, 0], [1, 0, 0]],
        [[2, 2, 1], [2, 1, 0]],
    ])

    dyepeps = np.array([
        [0, 0, 0],
        [1, 1, 10],
        [1, 2, 5],
        [2, 2, 5],
        [0, 3, 10],
    ])
    # fmt: on

    n_samples_per_pep = 5
    n_channels = 2
    n_cycles = 3
    n_peps = 4

    def it_removes_all_zero_rows():
        radiometry, true_pep_iz, true_dye_iz = sim_v2_worker._radmat_sim(
            dyemat,
            dyepeps,
            ch_params_with_noise,
            n_samples_per_pep,
            n_channels,
            n_cycles,
        )
        assert radiometry.shape == (10, n_channels, n_cycles)
        assert true_pep_iz.tolist() == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        assert not np.any(np.all(radiometry == 0.0, axis=(1, 2)))

    def it_returns_reasonable_radiometry():
        radiometry, true_pep_iz, true_dye_iz = sim_v2_worker._radmat_sim(
            dyemat,
            dyepeps,
            ch_params_with_noise,
            n_samples_per_pep,
            n_channels,
            n_cycles,
        )
        assert radiometry.shape == (10, n_channels, n_cycles)
        assert true_pep_iz.tolist() == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        # I'm not sure of a good test here
        # assert np.all(radiometry[radiometry > 0.0] > 1000.0)

    def it_returns_correct_radiometry_with_no_noise():
        # By using no noise, we can just compare that radiometry gave back the dyemat
        # but with each peptide repeated
        radiometry, true_pep_iz, true_dye_iz = sim_v2_worker._radmat_sim(
            dyemat,
            dyepeps,
            ch_params_no_noise,
            n_samples_per_pep,
            n_channels,
            n_cycles,
        )

        assert np.all(radiometry[0:5] == dyemat[1, :].astype(RadType),)

        assert np.all(
            (
                (radiometry[5:10] == dyemat[1, :].astype(RadType))
                | (radiometry[5:10] == dyemat[2, :].astype(RadType))
            ),
        )

        # fmt: off
        assert true_dye_iz[0:5].tolist() == [1, 1, 1, 1, 1]
        # fmt: on

        assert np.all((true_dye_iz[5:10] == 1) | (true_dye_iz[5:10] == 2))

        # fmt: off
        assert true_pep_iz.tolist() == [
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
        ]
        # fmt: on

    zest()


def zest_sim_v2_worker():
    prep_result = prep_fixtures.result_simple_fixture()

    def _sim(err_kwargs=None, _prep_result=None, sim_kwargs=None):
        if _prep_result is None:
            _prep_result = prep_result

        error_model = ErrorModel.no_errors(n_channels=2, **(err_kwargs or {}))

        sim_v2_params = SimV2Params.construct_from_aa_list(
            ["A", "B"], error_model=error_model, n_edmans=4, **(sim_kwargs or {})
        )

        return sim_v2_worker.sim_v2(sim_v2_params, _prep_result), sim_v2_params

    def it_returns_train_dyemat():
        # Because it has no errors, there's only a perfect dyemats
        sim_v2_result, sim_v2_params = _sim()
        assert sim_v2_result.train_dyemat.shape == (4, 5 * 2)  # 5 cycles, 2 channels
        assert utils.np_array_same(
            sim_v2_result.train_dyemat,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 2, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 2, 1, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        )

    def it_returns_train_dyemat_with_a_zero_row():
        sim_v2_result, sim_v2_params = _sim()
        assert np.all(sim_v2_result.train_dyemat[0, :] == 0)

    def it_returns_train_dyepeps():
        sim_v2_result, sim_v2_params = _sim()
        assert utils.np_array_same(
            sim_v2_result.train_dyepeps,
            np.array(
                [[0, 0, 0], [1, 1, 5000], [2, 2, 5000], [3, 3, 5000],], dtype=np.uint64
            ),
        )

    def it_handles_non_fluorescent():
        sim_v2_result, sim_v2_params = _sim(dict(p_non_fluorescent=0.5))
        # Check that every dyepep other than the nul-row
        # should have n_reads (col=2) a lot less than 5000.
        assert np.all(sim_v2_result.train_dyepeps[1:, 2] < 2000)

    def it_returns_no_all_dark_samples():
        sim_v2_result, sim_v2_params = _sim(dict(p_non_fluorescent=0.99))
        assert not np.any(sim_v2_result.train_dyepeps[1:, 0] == 0)

    def it_returns_recalls():
        sim_v2_result, sim_v2_params = _sim(dict(p_non_fluorescent=0.50))
        assert sim_v2_result.train_pep_recalls.shape[0] == 4  # 4 peps
        assert (
            sim_v2_result.train_pep_recalls[0] == 0.0
        )  # The nul record should have no recall
        assert np.all(
            sim_v2_result.train_pep_recalls[1:] < 0.85
        )  # The exact number is hard to say, but it should be < 1

    def it_emergency_escapes():
        sim_v2_result, sim_v2_params = _sim(dict(p_non_fluorescent=0.99))
        # When nothing is fluorescent, everything should have zero recall
        assert np.all(sim_v2_result.train_pep_recalls == 0.0)

    def it_handles_empty_dyepeps():
        sim_v2_result, sim_v2_params = _sim(dict(p_non_fluorescent=1.0))
        assert np.all(sim_v2_result.train_pep_recalls == 0.0)

    def decoys():
        prep_with_decoy = prep_fixtures.result_simple_fixture(has_decoy=True)
        sim_v2_result, sim_v2_params = _sim(dict(), prep_with_decoy)

        def it_maintains_decoys_for_train():
            assert sim_v2_result.train_dyemat.shape == (4, 10)

        def it_removes_decoys_for_test():
            # 1000 because the nul-dye track should be removed
            assert sim_v2_result.test_radmat.shape == (1000, 2, 5)

        zest()

    def it_skips_row_noise():
        sim_v2_result, sim_v2_params = _sim(err_kwargs=dict(row_k_sigma=0.0))
        assert np.all(sim_v2_result.test_true_ks == 1.0)

    def it_adds_row_noise():
        sim_v2_result, sim_v2_params = _sim(err_kwargs=dict(row_k_sigma=0.5))
        assert np.any(sim_v2_result.test_true_ks != 1.0)

    @zest.skip(reason="Not implemented")
    def it_raises_if_train_and_test_identical():
        raise NotImplementedError

    zest()
