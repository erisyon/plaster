from scipy import stats
import numpy as np
from plaster.run.sim_v2.sim_v2_result import RadType, DyeType
from plaster.run.nn_v2.c import nn_v2 as c_nn_v2
from plaster.run.sigproc_v2.sigproc_v2_fixtures import synthetic_radmat_from_dyemat
from plaster.run.error_model import GainModel
from plaster.tools.c_common.c_common_tools import CException
from zest import zest
from plaster.tools.log.log import debug


def zest_c_nn_v2():
    # fmt: off
    dyemat = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 1, 1],
        [3, 2, 1],
    ], dtype=DyeType)
    # fmt: on

    dyepeps = np.array(
        [
            # (dyt_i, pep_i, count)
            [1, 2, 30],
            [1, 1, 10],
            [2, 1, 10],
            [3, 1, 10],
        ],
        dtype=np.uint64,
    )

    from plaster.run.nn_v2.c import nn_v2

    nn_v2.init()

    gain_model = GainModel.test_fixture()

    def _test(
        radmat,
        n_neighbors=4,
        run_against_all_dyetracks=False,
        run_row_k_fit=True,
        _dyepeps=dyepeps,
    ):
        with c_nn_v2.context(
            dyemat,
            _dyepeps,
            radmat.astype(RadType),
            gain_model,
            n_neighbors=n_neighbors,
            run_row_k_fit=run_row_k_fit,
            run_against_all_dyetracks=run_against_all_dyetracks,
        ) as nn_v2_context:
            c_nn_v2.do_classify_radrows(nn_v2_context, 0, len(radmat))
            return nn_v2_context

    def it_catches_non_sequential_dyt_iz_in_dyepeps():
        _dyepeps = np.array([[1, 1, 10], [2, 1, 10], [1, 2, 30],], dtype=np.uint64,)
        radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dyemat(
            dyemat, gain_model, n_samples=5
        )
        with zest.raises(CException, in_args="Non sequential dyt_i"):
            _test(radmat, _dyepeps=_dyepeps)

    def it_enforces_reverse_sort_on_count_per_dyt():
        _dyepeps = np.array([[1, 1, 10], [1, 2, 30],], dtype=np.uint64,)
        radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dyemat(
            dyemat, gain_model, n_samples=5
        )
        with zest.raises(CException, in_args="must be reverse sorted by count per dyt"):
            _test(radmat, _dyepeps=_dyepeps)

    def it_classifies():
        radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dyemat(
            dyemat, gain_model, n_samples=5
        )
        nn_v2_context = _test(radmat)
        n_same = (true_dyt_iz == nn_v2_context.pred_dyt_iz).sum()
        assert n_same >= int(0.8 * true_dyt_iz.shape[0])

    def it_fits_k():
        gain_model = GainModel.test_fixture()
        gain_model.row_k_sigma = 0.2

        radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dyemat(
            dyemat, gain_model, n_samples=500
        )
        mask = true_dyt_iz > 0
        radmat = radmat[mask]
        true_ks = true_ks[mask]
        nn_v2_context = _test(radmat)

        # Uncomment this to compare to random
        # true_ks = np.random.normal(1.0, 0.5, true_ks.shape[0])

        # Check that there's a reasonable correlation between true and pred k
        # I ran this several times and found with random true_ks
        # ie: true_ks = np.random.normal(1.0, 0.5, true_ks.shape[0])
        # was like 0.02 where real was > 0.4
        pks = nn_v2_context.pred_ks
        mask = ~np.isnan(pks)
        pear_r, _ = stats.pearsonr(true_ks[mask], pks[mask])
        assert pear_r > 0.4

    def it_compares_to_all_dyetracks_without_row_fit():
        radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dyemat(
            dyemat, gain_model, n_samples=5
        )
        nn_v2_context = _test(
            radmat, n_neighbors=0, run_against_all_dyetracks=True, run_row_k_fit=False
        )

        # In this mode I expect to get back outputs for every radrow vs every dytrow

        assert (true_dyt_iz == nn_v2_context.pred_dyt_iz).sum() >= int(
            0.8 * true_dyt_iz.shape[0]
        )
        assert np.all(nn_v2_context.against_all_dyetrack_pred_ks[:, 1:] == 1.0)
        assert nn_v2_context.against_all_dyetrack_pvals.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )
        assert nn_v2_context.against_all_dyetrack_pred_ks.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )

    def it_compares_to_all_dyetracks_with_row_fit():
        gain_model = GainModel.test_fixture()
        gain_model.row_k_sigma = 0.2
        radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dyemat(
            dyemat, gain_model, n_samples=500
        )

        mask = true_dyt_iz > 0
        radmat = radmat[mask]
        true_ks = true_ks[mask]
        nn_v2_context = _test(
            radmat, n_neighbors=0, run_against_all_dyetracks=True, run_row_k_fit=True,
        )

        # Uncomment this to compare to random
        # true_ks = np.random.normal(1.0, 0.5, true_ks.shape[0])

        # In this mode I expect to get back outputs for every radrow vs every dytrow
        assert nn_v2_context.against_all_dyetrack_pvals.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )
        assert nn_v2_context.against_all_dyetrack_pred_ks.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )

        pks = nn_v2_context.pred_ks
        mask = ~np.isnan(pks)
        pear_r, _ = stats.pearsonr(true_ks[mask], pks[mask])
        assert pear_r > 0.5

    zest()
