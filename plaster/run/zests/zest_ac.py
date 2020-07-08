import numpy as np
import pandas as pd
from zest import zest
from plaster.run.sim_v1.sim_v1_params import SimV1Params
from plaster.run.error_model import ErrorModel
from plaster.run.nn import (
    _step_1_create_unit_radmat,
    _step_3_create_inverse_variances,
    _step_2_create_neighbors_lookup,
    _get_neighbor_iz,
    _do_nn_and_gmm,
    _step_5_mle_pred_dt_to_pep,
)
from plaster.tools.schema import check
from plaster.tools.log.log import debug


def _stub_sim_params(error_model, n_samples):
    return SimParams.construct_from_aa_list(
        ["A", "C"],
        error_model=error_model,
        n_samples=n_samples,
        n_pres=1,
        n_mocks=0,
        n_edmans=2,
    )


some_error_model = ErrorModel.no_errors(n_channels=2, sigma=0.16)


no_error_model = ErrorModel.no_errors(n_channels=2)


def _stub_dyemat_and_true():
    dyemat = np.array(
        [
            [[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]],
            [[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]],
            [[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.1, 0.1, 0.0], [0.1, 0.1, 0.1]],
        ]
    )

    true_pep_iz = np.array([1, 1, 1, 1, 3, 1, 3])

    return dyemat, true_pep_iz


def zest_step_1_create_unit_radmat():
    radmat = np.array(
        [[[20.0, 10.0, 0.0], [5.0, 5.0, 5.0]], [[30.0, 20.0, 10.0], [10.0, 5.0, 0.0]],]
    )
    channel_i_to_gain = np.array([10.0, 5.0])

    unit_radmat = _step_1_create_unit_radmat(channel_i_to_gain, radmat)

    expected = np.array(
        [[[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]], [[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]],]
    )

    assert np.allclose(unit_radmat, expected)


def zest_step_2_create_nearest_neighbors_lookup():
    dyemat, true_pep_iz = _stub_dyemat_and_true()

    (dt_mat, dyetracks_df, dt_pep_sources_df, flann) = _step_2_create_neighbors_lookup(
        dyemat, true_pep_iz
    )

    def it_creates_the_dt_mat():
        assert np.allclose(
            dt_mat,
            np.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.1, 0.1, 0.0], [0.1, 0.1, 0.1]],
                    [[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                    [[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]],
                ]
            ),
        )

    def it_adds_the_weights():
        assert dyetracks_df.weight.tolist() == [0, 1, 4, 2]

    def it_sets_the_sources():
        assert dt_pep_sources_df.pep_i.tolist() == [3, 1, 1, 3]
        assert dt_pep_sources_df.n_rows.tolist() == [1, 4, 1, 1]

    def it_gets_neighbors():
        x = np.array([3.1, 2.1, 1.1, 2.1, 1.1, 0.1])
        nn_iz, _ = flann.nn_radius(np.array(x), 1.0, max_nn=1)
        assert nn_iz == [3]

        neighbor_iz = _get_neighbor_iz(flann, x, n_neighbors=4, default=0, radius=20.0)
        assert neighbor_iz.tolist() == [3, 2, 1, 0]

    def it_adds_a_zero_record_first():
        assert np.all(dt_mat[0, :, :] == 0.0)

    zest()


def zest_step_3_create_inverse_variances():
    dyemat = np.array(
        [
            [[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    channel_i_to_vpd = np.array([2.0, 3.0])

    inv_var = _step_3_create_inverse_variances(dyemat, channel_i_to_vpd)

    expected = 1.0 / np.array(
        [
            [[8.0, 2.0, 0.5], [3.0, 3.0, 3.0]],
            [[18.0, 8.0, 2.0], [12.0, 3.0, 0.75]],
            [[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]],
        ]
    )

    assert np.allclose(inv_var, expected)


@zest.skip(reason="zbs fix please :)")
def zest_do_nn_and_gmm():
    dyemat, true_pep_iz = _stub_dyemat_and_true()

    dt_mat, _, _, flann = _step_2_create_neighbors_lookup(dyemat, true_pep_iz)

    x = np.array([[3.1, 2.1, 1.1], [2.1, 1.1, 0.1]])
    dyerow = np.array([[3, 2, 1], [2, 1, 0]])

    dt_weights = np.array([0.0, 1.0, 1.0, 1.0])

    def it_finds():
        _, pred, score, _ = _do_nn_and_gmm(
            x, dyerow, dt_mat, np.ones_like(dt_mat), dt_weights, flann, use_gmm=False
        )
        check.array_t(pred, shape=(1,))
        check.array_t(score, shape=(1,))
        assert pred.tolist() == [3]
        assert score[0] > 0.9

    zest()


@zest.skip(reason="zbs fix please :)")
def zest_step_5_mle_pred_dt_to_pep():
    pred_dt_iz = np.array([1, 2])
    dt_scores = np.array([0.5, 0.75])
    dt_pep_sources_df = pd.DataFrame(
        dict(
            dye_i=[1, 1, 1, 2, 2, 2, 4],
            pep_i=[1, 2, 3, 1, 2, 3, 5],
            n_rows=[2, 4, 9, 7, 8, 6, 1],
        )
    )

    pred_pep_iz, pep_scores, pred_scores = _step_5_mle_pred_dt_to_pep(
        pred_dt_iz, dt_scores, dt_pep_sources_df
    )

    assert pred_pep_iz.tolist() == [3, 2]
    assert pep_scores.tolist() == [9.0 / 15.0, 8.0 / 21.0]
    assert pred_scores.tolist() == [0.5 * 9.0 / 15.0, 0.75 * 8.0 / 21.0]
