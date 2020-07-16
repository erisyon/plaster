import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from plaster.run.nn_v1.nn_v1_params import NNV1Params
from plaster.run.sim_v1.sim_v1_result import (
    ArrayResult,
    DyeType,
    DyeWeightType,
    IndexType,
    RadType,
    RecallType,
    SimV1Result,
)
from plaster.run.sim_v1.sim_v1_params import SimV1Params, ErrorModel
from plaster.run.nn_v1 import nn_v1
from plaster.tools.utils import tmp
from plaster.vendor import pyflann
from zest import zest


def zest_nn_v1_step_1_create_neighbors_lookup():
    def _make_dyemat():
        n_peps = 3
        n_samples = 7
        n_channels = 2
        n_cycles = 5
        with tmp.tmp_folder(chdir=True):
            dyemat = ArrayResult(
                "dyemat",
                shape=(n_peps, n_samples, n_channels, n_cycles),
                dtype=DyeType,
                mode="w+",
            ).arr()
            dyemat[1, 0:5] = np.array([[2, 2, 1, 1, 0], [2, 1, 0, 0, 0],])
            dyemat[1, 5:7] = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0],])
            dyemat[2, 0:1] = np.array(
                [
                    # Same as dyemat[1][0:5]
                    [2, 2, 1, 1, 0],
                    [2, 1, 0, 0, 0],
                ]
            )
            dyemat[2, 1:7] = np.array(
                [
                    # Unique
                    [3, 3, 2, 2, 0],
                    [2, 1, 0, 0, 0],
                ]
            )
            # output_dt_mat is big enough to hold every possible dyetrack but would
            # be truncated after this call.
            output_dt_mat = ArrayResult(
                "dt_mat",
                shape=(n_peps * n_samples, n_channels, n_cycles),
                dtype=DyeType,
                mode="w+",
            ).arr()
            return dyemat, output_dt_mat

    def it_creates_lookup():
        dyemat, output_dt_mat = _make_dyemat()
        (
            dyetracks_df,
            dt_pep_sources_df,
            dye_to_best_pep_df,
            flann,
            n_dts,
        ) = nn_v1._step_1_create_neighbors_lookup_singleprocess(dyemat, output_dt_mat)

        def it_uniqifies_dyemat():
            assert n_dts == 4
            assert np.all(output_dt_mat[1] == [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
            assert np.all(output_dt_mat[2] == [[2, 2, 1, 1, 0], [2, 1, 0, 0, 0]])
            assert np.all(output_dt_mat[3] == [[3, 3, 2, 2, 0], [2, 1, 0, 0, 0]])

        def it_leaves_unused_rows_of_output_dt_mat_untouched():
            assert np.all(output_dt_mat[n_dts:] == 0)

        def it_includes_a_null_row():
            assert np.all(output_dt_mat[0] == [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

        def it_returns_dyetracks_df():
            # TODO: Rename dye_i to dt_i
            assert dyetracks_df.dye_i.values.tolist() == [0, 1, 2, 3]
            assert dyetracks_df.weight.values.tolist() == [7, 2, 6, 6]

        def it_returns_sources_df():
            # TODO: Reanme dye_i t dt_i
            assert dt_pep_sources_df.dye_i.values.tolist() == [0, 1, 2, 2, 3]
            assert dt_pep_sources_df.pep_i.values.tolist() == [0, 1, 1, 2, 2]
            assert dt_pep_sources_df.n_rows.values.tolist() == [7, 2, 5, 1, 6]

        def it_returns_flann():
            assert isinstance(flann, pyflann.index.FLANN)

        def it_computes_best_pep_df():
            assert (
                dye_to_best_pep_df.loc[0].pep_i == 0
                and dye_to_best_pep_df.loc[0].score == 1.0
            )
            assert (
                dye_to_best_pep_df.loc[1].pep_i == 1
                and dye_to_best_pep_df.loc[1].score == 1.0
            )
            assert (
                dye_to_best_pep_df.loc[2].pep_i == 1
                and dye_to_best_pep_df.loc[2].score == 5.0 / 6.0
            )
            assert (
                dye_to_best_pep_df.loc[3].pep_i == 2
                and dye_to_best_pep_df.loc[3].score == 1.0
            )

        zest()

    def it_raises_if_no_null_row():
        dyemat, output_dt_mat = _make_dyemat()
        with zest.raises(ValueError, in_args="No null row"):
            nn_v1._step_1_create_neighbors_lookup_singleprocess(
                dyemat[1:], output_dt_mat
            )

    @zest.skip("T", "TODO")
    def it_uniqifies_over_blocks():
        raise NotImplementedError

    zest()


def zest_nn_v1_step_1_create_neighbors_lookup_multiprocess():
    def it_gets_same_result_as_single_threaded():
        # n_peps, n_samples, n_channels, n_cycles = (50, 1000, 2, 15)
        n_peps, n_samples, n_channels, n_cycles = (20, 100, 2, 15)
        bin_vecs = np.random.randint(
            0, 2, size=(n_peps, n_samples, n_channels, n_cycles)
        )
        dyemat = np.cumsum(bin_vecs, axis=3)[:, :, :, ::-1]
        dyemat[0, 0] = np.zeros((n_channels, n_cycles), dtype=DyeType)
        dyemat = np.repeat(dyemat, 80, 0)
        np.random.shuffle(dyemat)

        with tmp.tmp_folder(chdir=True):
            output_dt_mat_st = ArrayResult(
                "dt_mat_st",
                shape=(n_peps * n_samples, n_channels, n_cycles),
                dtype=DyeType,
                mode="w+",
            ).arr()

            output_dt_mat_mt = ArrayResult(
                "dt_mat_mt",
                shape=(n_peps * n_samples, n_channels, n_cycles),
                dtype=DyeType,
                mode="w+",
            ).arr()

            # prof()
            (
                dyetracks_df_st,
                dt_pep_sources_df_st,
                dye_to_best_pep_df_st,
                flann_st,
                n_dts_st,
            ) = nn_v1._step_1_create_neighbors_lookup_singleprocess(
                dyemat, output_dt_mat_st
            )
            # prof("st")

            # prof()
            (
                dyetracks_df_mt,
                dt_pep_sources_df_mt,
                dye_to_best_pep_df_mt,
                flann_mt,
                n_dts_mt,
            ) = nn_v1._step_1_create_neighbors_lookup_multiprocess(
                dyemat, output_dt_mat_mt
            )
            # prof("mt")

            assert_frame_equal(dyetracks_df_st, dyetracks_df_mt)
            assert_frame_equal(dt_pep_sources_df_st, dt_pep_sources_df_st)
            assert_frame_equal(dye_to_best_pep_df_st, dye_to_best_pep_df_st)
            assert n_dts_st == n_dts_mt
            assert np.all(output_dt_mat_st == output_dt_mat_mt)

    zest()


def zest_nn_v1_step_2_create_inverse_variances():
    def it_scales_by_the_std():
        dt_mat = np.array(
            [[[0, 0, 0], [0, 0, 0]], [[2, 1, 0], [1, 1, 0]], [[1, 1, 0], [1, 0, 0]],]
        )
        channel_i_to_vpd = np.array([1.0, 4.0])  # 4 because it is a perfect square
        inv_var = nn_v1._step_2_create_inverse_variances(dt_mat, channel_i_to_vpd)
        assert np.all(
            # 0 becomes 0.5 * sqrt(1) == 0.5. 1.0 / 0.5**2
            # Channel 0:
            #   0.5 * sqrt(1) == 0.5... 1/0.5**2 == 4
            #   1.0 * sqrt(1) == 1.0... 1/1.0**2 == 1
            #   2.0 * sqrt(1) == 2.0... 1/2.0**2 == 0.25
            # Channel 1:
            #   0.5 * sqrt(4) == 1.0... 1/1.0**2 == 1
            #   1.0 * sqrt(4) == 2.0... 1/2.0**2 == 0.25
            inv_var
            == [
                [[4.0, 4.0, 4.0], [1.0, 1.0, 1.0],],
                [[0.25, 1.0, 4.0], [0.25, 0.25, 1.0],],
                [[1.0, 1.0, 4.0], [0.25, 1.0, 1.0],],
            ]
        )

    zest()


def zest_do_nn_v1():
    (
        nn_params,
        radmat,
        dt_mat,
        dt_inv_var_mat,
        dt_weights,
        flann,
        channel_i_to_gain_inv,
        dye_to_best_pep_df,
        dt_scores,
        scores,
        pred_pep_iz,
        pred_dt_iz,
        true_dt_iz,
        true_dyemat,
    ) = [None] * 14

    def _before():
        nonlocal nn_params, radmat, dt_mat, dt_inv_var_mat, dt_weights, flann
        nonlocal channel_i_to_gain_inv, dye_to_best_pep_df, dt_scores, scores
        nonlocal pred_pep_iz, pred_dt_iz, true_dt_iz, true_dyemat

        nn_params = NNV1Params()

        dt_mat = np.array(
            [
                [[0, 0, 0], [0, 0, 0]],  # Target 0
                [[2, 1, 0], [2, 2, 0]],  # Target 1
                [[1, 1, 0], [1, 0, 0]],  # Target 2
            ],
            dtype=DyeType,
        )

        dt_weights = np.array([0, 5, 10], dtype=DyeWeightType)

        true_dyemat = np.array(
            [
                [[1, 1, 0], [1, 0, 0]],  # Target == 2
                [[2, 1, 0], [2, 2, 0]],  # Target == 1
                [[10, 10, 9], [10, 10, 10]],  # Target == None
            ],
            dtype=DyeType,
        )
        radmat = np.array(
            [
                [[1.1, 0.9, 0.0], [1.1, 0.1, 0.0]],  # Target == 2
                [[2.1, 1.1, 0.0], [2.1, 1.9, 0.0]],  # Target == 1
                [[10.0, 10.0, 9.0], [10.0, 10.0, 10.0]],  # Target == None
            ],
            dtype=RadType,
        )

        channel_i_to_vpd = np.array([1.5, 2.0], dtype=RadType)

        channel_i_to_gain = np.array([10.0, 100.0], dtype=RadType)
        radmat = radmat * channel_i_to_gain[None, :, None]
        channel_i_to_gain_inv = 1.0 / channel_i_to_gain

        dt_inv_var_mat = nn_v1._step_2_create_inverse_variances(
            dt_mat, np.array(channel_i_to_vpd)
        )

        flann = nn_v1._create_flann(dt_mat)

        dye_to_best_pep_df = pd.DataFrame(
            dict(dye_i=[0, 1, 2], pep_i=[0, 2, 1], score=[1.0, 0.5, 1.0],)
        )

        n_rows = radmat.shape[0]
        with tmp.tmp_folder(chdir=True):
            dt_scores = ArrayResult(
                "dt_scores", nn_v1.ScoreType, (n_rows,), mode="w+"
            ).arr()
            scores = ArrayResult("scores", nn_v1.ScoreType, (n_rows,), mode="w+").arr()
            pred_pep_iz = ArrayResult(
                "pred_pep_iz", IndexType, (n_rows,), mode="w+"
            ).arr()
            pred_dt_iz = ArrayResult(
                "pred_dt_iz", IndexType, (n_rows,), mode="w+"
            ).arr()
            true_dt_iz = ArrayResult(
                "true_dt_iz", IndexType, (n_rows,), mode="w+"
            ).arr()

    def _run(i):
        nn_v1._do_nn(
            i,
            nn_params,
            radmat,
            dt_mat,
            dt_inv_var_mat,  # Inv variance of each target
            dt_weights,  # Weight of each target
            flann,  # Neighbor lookup index
            channel_i_to_gain_inv,  # Normalization term for each channel (radmat->unit_radmat)
            score_normalization=1.0,
            dye_to_best_pep_df=dye_to_best_pep_df,
            output_pred_dt_scores=dt_scores,
            output_pred_scores=scores,
            output_pred_pep_iz=pred_pep_iz,
            output_pred_dt_iz=pred_dt_iz,
            output_true_dt_iz=true_dt_iz,
            true_dyemat=true_dyemat,
        )

    def it_assigns_peptide_score():
        _run(1)
        assert (scores[1] / dt_scores[1]) == 0.5
        assert pred_pep_iz[1] == 2

    def it_classifies_correctly():
        # nonlocal channel_i_to_gain_inv, radmat, channel_i_to_gain_inv

        i = 0
        _run(i)
        assert pred_dt_iz[i] == 2

        i = 1
        _run(i)
        assert pred_dt_iz[i] == 1

        i = 2
        _run(i)
        assert pred_dt_iz[i] == 0

    def it_finds_true_dyemats():
        i = 0
        _run(i)
        assert true_dt_iz[i] == 2

        i = 1
        _run(i)
        assert true_dt_iz[i] == 1

        i = 2
        _run(i)
        assert true_dt_iz[i] == 0

    def it_copes_with_no_true_dyemats():
        nonlocal true_dyemat
        true_dyemat = None
        _run(1)
        assert np.all(true_dt_iz == 0)

    def it_copes_with_no_neighbors_found():
        _run(2)
        assert pred_dt_iz[2] == 0
        assert dt_scores[2] == 0.0
        assert true_dt_iz[2] == 0

    # def it_filters_low_weight_dyetrack():
    #     raise NotImplementedError
    #
    # def it_returns_gmm_winner():
    #     raise NotImplementedError
    #
    # def it_applies_rare_penalty():
    #     raise NotImplementedError
    #
    # def it_sets_output_arrays():
    #     raise NotImplementedError

    zest()


def zest_nn_v1():
    def it_sets_all_output_arrays():
        n_peps, n_samples, n_channels, n_cycles = (3, 2, 2, 3)
        nn_params = NNV1Params()
        sim_params = SimV1Params.construct_from_aa_list(
            ["A", "B"], error_model=ErrorModel.no_errors(n_channels)
        )
        sim_params.error_model.dyes[0].gain = 100.0
        sim_params.error_model.dyes[1].gain = 400.0
        sim_params._build_join_dfs()

        with tmp.tmp_folder(chdir=True):
            train_dyemat = ArrayResult(
                "train_dyemat",
                shape=(n_peps, n_samples, n_channels, n_cycles),
                dtype=DyeType,
                mode="w+",
            )
            train_dyemat[:] = np.array(
                [
                    [  # Pep 0
                        [[0, 0, 0], [0, 0, 0],],  # Sample 0
                        [[0, 0, 0], [0, 0, 0],],  # Sample 1
                    ],
                    [  # Pep 1
                        [[2, 2, 1], [1, 0, 0],],  # Sample 0
                        [[2, 2, 1], [1, 0, 0],],  # Sample 1
                    ],
                    [  # Pep 2
                        [[2, 2, 2], [2, 1, 0],],  # Sample 0
                        [  # Sample 1
                            [2, 2, 1],
                            [1, 0, 0],  # Same same sample 0 & 1 of pep 1
                        ],
                    ],
                ]
            )

            sim_result = SimV1Result(
                params=sim_params,
                train_dyemat=train_dyemat.arr(),
                # None of the following are used by nn
                train_radmat=ArrayResult(
                    "train_radmat", shape=(1,), dtype=RadType, mode="w+"
                ).arr(),
                train_recalls=ArrayResult(
                    "train_recalls", shape=(1,), dtype=RecallType, mode="w+"
                ).arr(),
                train_flus=ArrayResult(
                    "train_flus", shape=(1,), dtype=DyeType, mode="w+"
                ).arr(),
                train_flu_remainders=ArrayResult(
                    "train_flu_remainders", shape=(1,), dtype=DyeType, mode="w+"
                ).arr(),
            )

            test_radmat = ArrayResult(
                "test_radmat", shape=(3, n_channels, n_cycles), dtype=RadType, mode="w+"
            )
            test_radmat[:] = np.array(
                [
                    [  # pep 1, sample 0 & 1; pep 2, sample 1
                        [2.1, 1.9, 1.1],
                        [
                            0.9,
                            0.1,
                            0.1,
                        ],  # Should pred to dt 1, could be pep 1 or pep 2 but pep 1 has more instances
                    ],
                    [  # pep 0, sample 0
                        [0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1],  # Should pred to dt 0, must be pep 0
                    ],
                    [  # Pep 2, sample 0
                        [2.1, 1.9, 1.9],
                        [2.1, 1.1, 0.1],  # Should pred to dt 2, must be pep 2
                    ],
                ]
            )
            test_radmat[:, 0, :] *= sim_params.error_model.dyes[0].gain
            test_radmat[:, 1, :] *= sim_params.error_model.dyes[1].gain

            nn_result = nn_v1.nn(nn_params, sim_result, test_radmat.arr())

            assert np.all(
                nn_result.dt_mat.arr()
                == [
                    [[0, 0, 0], [0, 0, 0]],
                    [[2, 2, 1], [1, 0, 0]],
                    [[2, 2, 2], [2, 1, 0]],
                ]
            )

            assert np.all(nn_result.dyetracks_df.dye_i.values == [0, 1, 2])
            assert np.all(nn_result.dyetracks_df.weight.values == [2, 3, 1])

            assert np.all(nn_result.dt_pep_sources_df.dye_i.values == [0, 1, 1, 2])
            assert np.all(nn_result.dt_pep_sources_df.pep_i.values == [0, 1, 2, 2])
            assert np.all(nn_result.dt_pep_sources_df.n_rows.values == [2, 2, 1, 1])

            assert np.all(nn_result.pred_dt_iz.arr() == [1, 0, 2])

            # TODO: Check all the nn_results here
            # Then I need to implement the avoidance of the max calc
            # And then I can profile it on large datasets
            assert np.all(nn_result.pred_pep_iz.arr() == [1, 0, 2])

            assert np.all(
                (0 <= nn_result.scores.arr()) & (nn_result.scores.arr() <= 1.0)
            )
            assert nn_result.scores.shape == (3,)

            assert np.all(
                (0 <= nn_result.dt_scores.arr()) & (nn_result.dt_scores.arr() <= 1.0)
            )
            assert nn_result.dt_scores.shape == (3,)

    zest()
