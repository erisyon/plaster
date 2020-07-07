from plumbum import local
import numpy as np
import pandas as pd
from zest import zest
from plaster.run.sim import sim_worker
from plaster.run.sim.sim_params import SimParams
from plaster.run.prep.prep_result import PrepResult
from plaster.run.base_result import ArrayResult
from plaster.run.error_model import ErrorModel
from plaster.tools.log.log import debug
from plaster.tools.utils.utils import npf, np_array_same
from plaster.tools.utils import tmp


def _stub_sim_params(error_model, n_samples):
    return SimParams.construct_from_aa_list(
        ["A", "C"],
        error_model=error_model,
        n_samples_train=n_samples,
        n_samples_test=n_samples,
        n_pres=1,
        n_mocks=0,
        n_edmans=2,
    )


some_error_model = ErrorModel.no_errors(n_channels=2, sigma=0.16)


no_error_model = ErrorModel.no_errors(n_channels=2)


def zest_step_1_create_flu():
    sim_params = SimParams.construct_from_aa_list(
        ("A", "C"), error_model=ErrorModel.no_errors(n_channels=2),
    )

    pep_seq_df = pd.DataFrame(
        [(1, "A", 0), (1, "B", 1), (1, "C", 2),], columns=PrepResult.pep_seqs_columns
    )

    def it_return_flu_and_p_bright():
        flu, p_bright = sim_worker._step_1_create_flu_and_p_bright(
            pep_seq_df, sim_params
        )
        assert np.all(flu == [[1, 0, 0,], [0, 0, 1,],])
        assert np.all(p_bright == [[1, 0, 0,], [0, 0, 1,],])

    def it_extends_short_peptides():
        sim_params = SimParams.construct_from_aa_list(
            ("A", "C"), error_model=ErrorModel.no_errors(n_channels=2), n_edmans=6,
        )
        flu, p_bright = sim_worker._step_1_create_flu_and_p_bright(
            pep_seq_df, sim_params
        )
        assert flu.shape == (2, 7)
        assert np.all(flu[:, 3:] == 0.0)

    def it_computes_p_bright_from_product():
        error_model = ErrorModel.no_errors(n_channels=2)
        error_model.labels[0].p_failure_to_bind_amino_acid = 0.9
        error_model.labels[1].p_failure_to_bind_amino_acid = 0.8
        error_model.labels[0].p_failure_to_attach_to_dye = 0.7
        error_model.labels[1].p_failure_to_attach_to_dye = 0.6
        error_model.dyes[0].p_non_fluorescent = 0.5
        error_model.dyes[1].p_non_fluorescent = 0.4

        sim_params = SimParams.construct_from_aa_list(
            ("A", "C"), error_model=error_model, n_edmans=3
        )
        flu, p_bright = sim_worker._step_1_create_flu_and_p_bright(
            pep_seq_df, sim_params
        )
        assert flu.shape == (2, 4)
        expected = np.array(
            [
                [((1 - 0.9) * (1 - 0.7) * (1 - 0.5)), 0, 0, 0,],
                [0, 0, ((1 - 0.8) * (1 - 0.6) * (1 - 0.4)), 0,],
            ]
        )
        assert np.allclose(p_bright, expected)

    def it_labels_in_tail():
        sim_params = SimParams.construct_from_aa_list(
            ("A", "C"), error_model=ErrorModel.no_errors(n_channels=2), n_edmans=6,
        )
        pep_seq_df = pd.DataFrame(
            [
                (1, "B", 0),
                (1, "B", 1),
                (1, "B", 2),
                (1, "B", 3),
                (1, "A", 4),
                (1, "C", 5),
            ],
            columns=PrepResult.pep_seqs_columns,
        )
        flu, p_bright = sim_worker._step_1_create_flu_and_p_bright(
            pep_seq_df, sim_params
        )
        assert np.all(flu == [[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],])
        assert np.all(p_bright == [[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],])

    zest()


def zest_step_2_initialize_samples_including_dark_sampling():
    flu = np.array([[1, 0, 0], [0, 1, 0],])
    p_bright = np.array([[0.5, 0, 0], [0, 0.95, 0],])

    n_samples = 5

    def it_copies():
        with zest.mock(sim_worker._rand3, returns=np.full((n_samples, 2, 3), 1.0)):
            sim_worker._step_2_initialize_samples_including_dark_sampling(
                flu, p_bright, n_samples=n_samples
            )
            assert flu[0][0] == 1  # Check that the original flu was not modified.

    def it_darkens():
        # Large random numbers mean it will go dark so a 0.9 will
        # darken the 0.5 but not the 1.0
        with zest.mock(sim_worker._rand3, returns=np.full((n_samples, 2, 3), 0.9)):
            flu_samples = sim_worker._step_2_initialize_samples_including_dark_sampling(
                flu, p_bright, n_samples=n_samples
            )
            assert flu_samples.shape == (n_samples, 2, 3)
            assert np.all(flu_samples[:, 1, 1] == 1.0)
            assert np.all(flu_samples[:, 0, 0] == 0.0)

    def it_does_not_darken():
        # Similarly a low number will keep everything lit
        with zest.mock(sim_worker._rand3, returns=np.full((n_samples, 2, 3), 0.1)):
            flu_samples = sim_worker._step_2_initialize_samples_including_dark_sampling(
                flu, p_bright, n_samples=n_samples
            )
            assert flu_samples.shape == (n_samples, 2, 3)
            assert np.all(flu_samples[:, 1, 1] == 1.0)
            assert np.all(flu_samples[:, 0, 0] == 1.0)

    zest()


def zest_step_3a_cycle_edman():
    n = np.nan
    samples = npf([[[1, 1, 0], [0, 1, 0]], [[0, 1, 1], [0, 0, 1]],])

    def it_can_edman_fail():
        # Failure means Edman did not happen; p_edman_failure=1.0 and we no change
        result = sim_worker._step_3a_cycle_edman(
            samples, is_mock=False, p_edman_failure=1.0
        )
        assert np_array_same(samples, result)

    def it_can_edman_degrade():
        # Degrade = Edman success; thus p_edman_failure=0.0
        result = sim_worker._step_3a_cycle_edman(
            samples, is_mock=False, p_edman_failure=0.0
        )
        expected = npf([[[n, 1, 0], [n, 1, 0]], [[n, 1, 1], [n, 0, 1]],])
        assert np_array_same(expected, result)

        def it_makes_a_copy():
            # The original should not have changed
            assert samples[0, 0, 0] == 1.0

        zest()

    def it_degrades_at_first_non_nan():
        samples = npf([[[n, 1, 0], [n, 1, 0]], [[0, 1, 1], [0, 0, 1]],])
        result = sim_worker._step_3a_cycle_edman(
            samples, is_mock=False, p_edman_failure=0.0
        )
        expected = npf([[[n, n, 0], [n, n, 0]], [[n, 1, 1], [n, 0, 1]],])
        assert np_array_same(expected, result)

    def it_can_mock():
        # A mock cycle is the same as a edman_failure no matter what (p_edman_failure=0.0)
        # and yet I should still see failure behavior
        result = sim_worker._step_3a_cycle_edman(
            samples, is_mock=True, p_edman_failure=0.0
        )
        assert np_array_same(samples, result)

    def it_can_degrade_one_sample_and_not_another():
        samples = npf([[[n, 1, 0], [n, 1, 0]], [[0, 1, 1], [0, 0, 1]],])
        n_samples = 2
        with zest.mock(sim_worker._rand1, returns=np.full((n_samples,), [0.0, 1.0])):
            result = sim_worker._step_3a_cycle_edman(
                samples, is_mock=False, p_edman_failure=0.5
            )
            expected = npf([[[n, 1, 0], [n, 1, 0]], [[n, 1, 1], [n, 0, 1]],])
            assert np_array_same(expected, result)

    zest()


def zest_step_3b_photobleach():
    n = np.nan
    samples = npf([[[1, 1, 0], [0, 1, 0]], [[n, 1, 1], [n, 0, 1]],])

    def it_bleaches_all():
        result = sim_worker._step_3b_photobleach(
            samples, p_bleach_per_cycle_by_channel=[1.0, 1.0]
        )
        expected = npf([[[0, 0, 0], [0, 0, 0]], [[n, 0, 0], [n, 0, 0]],])
        assert np_array_same(expected, result)

        def it_makes_a_copy():
            # The original should not have changed
            assert samples[0, 0, 0] == 1.0

        zest()

    def it_bleaches_channels_independently():
        result = sim_worker._step_3b_photobleach(
            samples, p_bleach_per_cycle_by_channel=[1.0, 0.0]
        )
        expected = npf([[[0, 0, 0], [0, 1, 0]], [[n, 0, 0], [n, 0, 1]],])
        assert np_array_same(expected, result)

    zest()


def zest_step_3c_detach():
    n = np.nan
    samples = npf([[[1, 1, 0], [0, 1, 0]], [[n, 1, 1], [n, 0, 1]],])

    def it_detach():
        result = sim_worker._step_3c_detach(samples, 1.0)
        expected = np.full_like(samples, n)
        assert np_array_same(expected, result)

        def it_make_a_copy():
            # The original should not have changed
            assert samples[0, 0, 0] == 1.0

        zest()

    def it_fail_to_detach():
        result = sim_worker._step_3c_detach(samples, 0.0)
        assert np_array_same(samples, result)

    zest()


def zest_step_3_evolve_cycles():

    # fmt: off
    samples = npf([
        [[1, 1, 0], [0, 1, 0]],
        [[0, 1, 1], [0, 0, 1]],
    ])
    # fmt: on

    sim_params = None

    def _before():
        nonlocal sim_params
        sim_params = SimParams.construct_from_aa_list(
            ["DE", "C"],
            error_model=ErrorModel.no_errors(n_channels=2, beta=7500.0),
            n_pres=0,
            n_mocks=1,
            n_edmans=1,
        )

    def it_proceeds_with_no_error():
        evolution = sim_worker._step_3_evolve_cycles(samples, sim_params)

        # fmt: off
        n = np.nan
        expected = npf([
            [
                [[1, 1, 0], [0, 1, 0]],
                [[0, 1, 1], [0, 0, 1]],
            ],
            [
                [[n, 1, 0], [n, 1, 0]],
                [[n, 1, 1], [n, 0, 1]],
            ],
        ])
        # fmt: on

        if not np_array_same(evolution, expected):
            debug(evolution)
            debug(sim_params)

        assert np_array_same(evolution, expected)

    def it_proceeds_with_errors():
        # Example error: detach 100%
        sim_params.error_model.p_detach = 1.0
        sim_params._build_join_dfs()

        evolution = sim_worker._step_3_evolve_cycles(samples, sim_params)
        expected = np.full_like(evolution, np.nan)

        assert np_array_same(evolution, expected)

        def it_makes_a_copy():
            assert samples[0, 0, 0] == 1.0

        zest()

    zest()


def zest_step_4_make_dyemat():
    n = np.nan

    def it_sums():
        # fmt: off

        # Using a huge array here to make sure all my axes
        # are getting set properly

        evolution = npf([
            [
                [[1, 1, 0], [0, 1, 0]],
                [[0, 1, 1], [0, 0, 1]],
                [[1, 1, 1], [1, 0, 1]],
                [[1, 1, 1], [1, 0, 1]],
            ],
            [
                [[n, 1, 0], [n, 1, 0]],
                [[n, 1, 1], [n, 0, 1]],
                [[n, n, 1], [n, n, 1]],
                [[n, 1, 1], [n, 0, 1]],
            ],
            [
                [[n, 1, 0], [n, 1, 0]],
                [[n, 1, 1], [n, 0, 1]],
                [[n, n, 1], [n, n, 1]],
                [[n, 1, 1], [n, 0, 1]],
            ],
            [
                [[n, 1, 0], [n, 1, 0]],
                [[n, 1, 1], [n, 0, 1]],
                [[n, n, 1], [n, n, 1]],
                [[n, 1, 1], [n, 0, 1]],
            ],
            [
                [[n, 3, 0], [n, 4, 0]],
                [[n, 1, 5], [n, 0, 1]],
                [[n, n, 9], [n, n, 1]],
                [[n, 1, 1], [n, 0, 2]],
            ],
        ])

        dyemat = sim_worker._step_4_make_dyemat(evolution)

        # The axes are now rearranged so that there's
        # 4 samples, 2 channels, 5 cycles
        # I've thrown enough noise in to make sure I've got
        # the dimensions as I want

        expected = npf([
            [
                [2, 1, 1, 1, 3],
                [1, 1, 1, 1, 4],
            ],
            [
                [2, 2, 2, 2, 6],
                [1, 1, 1, 1, 1],
            ],
            [
                [3, 1, 1, 1, 9],
                [2, 1, 1, 1, 1],
            ],
            [
                [3, 2, 2, 2, 2],
                [2, 1, 1, 1, 2],
            ],
        ])
        # fmt: on

        assert np_array_same(dyemat, expected)

    def it_zeros_nans():
        # fmt: off
        evolution = npf([
            [
                [[n, n, n], [n, n, n]],
                [[n, n, n], [n, n, n]],
            ],
        ])

        dyemat = sim_worker._step_4_make_dyemat(evolution)

        expected = npf([
            [
                [0, 0],
                [0, 0],
            ],
        ])
        # fmt: on
        assert np_array_same(dyemat, expected)

    zest()


def zest_step_5_make_radmat():
    # fmt: off
    dyemat = npf([
        [
            [2, 1],
            [2, 1],
        ],
        [
            [0, 0],
            [0, 0],
        ],
    ])

    sim_params = SimParams.construct_from_aa_list(
        ["DE", "C"],
        error_model=ErrorModel.no_errors(n_channels=2, beta=7500.0, sigma=0.16),
        n_pres=1,
        n_mocks=0,
        n_edmans=1,
    )

    radmat = sim_worker._step_5_make_radmat(dyemat, sim_params)

    def it_makes_radmat():
        assert np.all( radmat[0] > 4000.0 )

    def it_deals_with_zeros():
        assert np.all( radmat[1] == 0.0 )

    # fmt: on
    zest()


def zest_step_6_compact_flu():
    flu = npf([[0, 0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 1],])

    compact_flu, remainder_flu = sim_worker._step_6_compact_flu(
        flu, n_edmans=4, n_channels=2
    )

    n = np.nan

    def it_makes_a_flu():
        assert np_array_same(compact_flu, [1, n, 0, 1])

    def it_makes_remainder():
        assert np_array_same(remainder_flu, [1, 2])

    zest()


def zest_do_pep_sim():
    sim_params = _stub_sim_params(no_error_model, 100)
    dyemat, radmat, recall = None, None, None

    def _make_arrays(name, n_peps, n_samples):
        dyemat = ArrayResult(
            f"{name}_dyemat",
            shape=(n_peps, n_samples, sim_params.n_channels, sim_params.n_cycles),
            dtype=np.uint8,
            mode="w+",
        )
        radmat = ArrayResult(
            f"{name}_radmat",
            shape=(n_peps, n_samples, sim_params.n_channels, sim_params.n_cycles),
            dtype=np.float32,
            mode="w+",
        )
        recall = ArrayResult(
            f"{name}_recall", shape=(n_peps,), dtype=np.float32, mode="w+",
        )
        return dyemat, radmat, recall

    def it_returns_no_all_dark_samples_on_valid_peps():
        with tmp.tmp_folder(chdir=True):
            prep_result = PrepResult.stub_prep_result(
                pros=[".", "ABCDEFGHI"],
                pro_is_decoys=[False, False],
                peps=[".", "AAA"],
                pep_pro_iz=[0, 1],
            )

            pep_seq_df = prep_result.pepseqs()
            n_samples = 1000
            dyemat, radmat, recall = _make_arrays(
                "test1", n_peps=2, n_samples=n_samples
            )
            sim_worker._do_pep_sim(
                pep_seq_df[pep_seq_df.pep_i == 1],
                sim_params,
                n_samples=n_samples,
                output_dyemat=dyemat,
                output_radmat=radmat,
                output_recall=recall,
            )
            assert not np.any(np.all(dyemat[1] == 0, axis=(1, 2)))

    def it_returns_the_fraction_of_all_dark_samples():
        with tmp.tmp_folder(chdir=True):
            n_samples = 5000
            sim_params = _stub_sim_params(
                ErrorModel.from_defaults(n_channels=2), n_samples
            )
            prep_result = PrepResult.stub_prep_result(
                pros=[".", "ABCDEFGHI"],
                pro_is_decoys=[False, False],
                peps=[".", "ABB"],
                pep_pro_iz=[0, 1],
            )

            pep_seq_df = prep_result.pepseqs()

            dyemat, radmat, recall = _make_arrays(
                "test1", n_peps=2, n_samples=n_samples
            )
            sim_worker._do_pep_sim(
                pep_seq_df[pep_seq_df.pep_i == 1],
                sim_params,
                n_samples=n_samples,
                output_dyemat=dyemat,
                output_radmat=radmat,
                output_recall=recall,
            )
            assert np.all((0.9 < recall[1]) & (recall[1] < 1.0))
            # 0.9 because it seems unlikely that in 5000 attempts on a 0.95% call
            # that you'd ever be off by that much.

    def it_gives_up_on_hard_peptides_and_returns_none():
        with tmp.tmp_folder(chdir=True):
            prep_result = PrepResult.stub_prep_result(
                pros=[".", "ABCDEFGHI"],
                pro_is_decoys=[False, False],
                peps=[".", "DDD"],
                pep_pro_iz=[0, 1],
            )

            pep_seq_df = prep_result.pepseqs()

            n_samples = 1000
            dyemat, radmat, recall = _make_arrays(
                "test1", n_peps=2, n_samples=n_samples
            )
            sim_worker._do_pep_sim(
                pep_seq_df[pep_seq_df.pep_i == 1],
                sim_params,
                n_samples=n_samples,
                output_dyemat=dyemat,
                output_radmat=radmat,
                output_recall=recall,
            )
            assert np.all(recall[:] == 0.0)

    zest()


def zest_sim():
    prep_result = PrepResult.stub_prep_result(
        pros=[".", "ABCDEFGHI", "ABC"],
        pro_is_decoys=[False, False, True],
        peps=[".", "ABC", "DAF", "ACH", "ABC"],
        pep_pro_iz=[0, 1, 1, 1, 2],
    )

    n_samples = 8
    n_peptides = 5
    n_channels = 2
    n_cycles = 3  # mock + edman (See below)

    def it_maintains_decoys_for_train():
        with tmp.tmp_folder(chdir=True):
            sim_params = _stub_sim_params(some_error_model, n_samples)
            sim_result = sim_worker.sim(sim_params, prep_result)
            assert sim_result.train_dyemat.shape == (
                n_peptides,
                n_samples,
                n_channels,
                n_cycles,
            )

    def it_removes_decoys_for_test():
        with tmp.tmp_folder(chdir=True):
            sim_params = _stub_sim_params(some_error_model, n_samples)
            sim_result = sim_worker.sim(sim_params, prep_result)
            assert sim_result.test_dyemat.shape == (
                n_peptides,
                n_samples,
                n_channels,
                n_cycles,
            )
            assert np.all(sim_result.test_dyemat[0] == 0)  # Nul should be all zero
            assert np.all(sim_result.test_dyemat[4] == 0)  # Decoy should be all zero
            assert sim_result.test_radmat.dtype == np.float32

    def it_raises_if_train_and_test_identical():
        with tmp.tmp_folder(chdir=True):
            with zest.raises(in_message="are identical"):
                sim_params = _stub_sim_params(no_error_model, n_samples)
                sim_worker.sim(sim_params, prep_result)

    def it_drop_all_darks():
        with tmp.tmp_folder(chdir=True):
            prep_result = PrepResult.stub_prep_result(
                pros=[".", "DD", "EE"],
                pro_is_decoys=[False, False, False],
                peps=[".", "DD", "EE"],
                pep_pro_iz=[0, 1, 2],
            )
            n_peptides = 3
            sim_params = _stub_sim_params(no_error_model, n_samples)
            sim_result = sim_worker.sim(sim_params, prep_result)
            assert sim_result.test_dyemat.shape == (
                n_peptides,
                n_samples,
                n_channels,
                n_cycles,
            )
            assert sim_result.test_dyemat.dtype == np.uint8
            assert np.all(sim_result.test_dyemat[:] == 0)  # All dark

            assert sim_result.train_dyemat.shape == (
                n_peptides,
                n_samples,
                n_channels,
                n_cycles,
            )
            assert sim_result.train_dyemat.dtype == np.uint8
            assert np.all(sim_result.train_recalls[:] == 0.0)

    def it_generates_flu_info():
        with tmp.tmp_folder(chdir=True):
            prep_result = PrepResult.stub_prep_result(
                pros=[".", "XAXCD", "XAXCDXX", "XCCXX"],
                pro_is_decoys=[False, False, False, False],
                peps=[".", "XAXCD", "XAXCDXX", "XCCXX"],
                pep_pro_iz=[0, 1, 2, 3],
            )
            sim_params = _stub_sim_params(some_error_model, n_samples)
            sim_result = sim_worker.sim(sim_params, prep_result)
            sim_result._generate_flu_info(prep_result)

            def it_computes_head_and_tail():
                _flus = sim_result._flus
                assert np.all(_flus[_flus.pep_i.isin([1, 2])].flu_count == 2)
                assert np.all(_flus[_flus.pep_i.isin([1, 2])].n_head_ch_0 == 1)
                assert np.all(_flus[_flus.pep_i.isin([1, 2])].n_head_ch_1 == 0)
                assert np.all(_flus[_flus.pep_i.isin([1, 2])].n_tail_ch_0 == 0)
                assert np.all(_flus[_flus.pep_i.isin([1, 2])].n_tail_ch_1 == 1)
                assert np.all(_flus[_flus.pep_i == 3].flu_count == 1)

            def it_peps__flus():
                df = sim_result.peps__flus(prep_result)
                assert "flustr" in df
                assert len(df) == 4

            def it_peps__flus__unique_flus():
                df = sim_result.peps__flus__unique_flus(prep_result)
                assert np.all(df.pep_i.values == [0, 3])

            zest()

    def it_surveys():
        with tmp.tmp_folder(chdir=True):
            n_samples = 1
            sim_params = _stub_sim_params(some_error_model, n_samples)
            sim_params.is_survey = True
            sim_params.n_samples_train = n_samples
            sim_params.n_samples_test = None
            sim_result = sim_worker.sim(sim_params, prep_result)
            assert sim_result.train_dyemat.shape == (
                n_peptides,
                n_samples,
                n_channels,
                n_cycles,
            )
            assert sim_result.train_dyemat.dtype == np.uint8
            assert sim_result.test_dyemat is None

    zest()
