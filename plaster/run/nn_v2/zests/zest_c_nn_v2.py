import numpy as np
from plaster.run.sim_v2.sim_v2_result import RadType, DyeType
from plaster.run.nn_v2.c import nn_v2 as c_nn_v2
from scipy.stats import norm
from zest import zest
from plaster.tools.log.log import debug


def sample_gaussian(beta, sigma, n_samples):
    return norm(beta, sigma).rvs(n_samples)


def _radmat_from_dyemat(dyemat, gain_model, n_samples, k_sigma=0.0):
    n_dyts, n_cols = dyemat.shape
    radmat = np.zeros((n_dyts * n_samples, n_cols))
    true_dyt_iz = np.zeros((n_dyts * n_samples,), dtype=int)
    for dyt_i, dyt in enumerate(dyemat):
        dyt_radmat = np.zeros((n_samples, n_cols))
        for col_i, dye_count in enumerate(dyt):
            if dye_count > 0:
                dyt_radmat[:, col_i] = np.exp(
                    sample_gaussian(
                        np.log(gain_model[0] * dye_count), gain_model[1], n_samples
                    )
                )
            else:
                dyt_radmat[:, col_i] = sample_gaussian(
                    gain_model[2], gain_model[3], n_samples
                )

        radmat[dyt_i * n_samples : (dyt_i + 1) * n_samples, :] = dyt_radmat

        true_dyt_iz[dyt_i * n_samples : (dyt_i + 1) * n_samples] = dyt_i

    n_radrows = radmat.shape[0]
    true_ks = np.ones((n_radrows,))
    if k_sigma > 0.0:
        true_ks = sample_gaussian(1.0, k_sigma, n_radrows)
        radmat = radmat * true_ks[:, None]

    return radmat, true_dyt_iz, true_ks


def zest_c_nn_v2():
    dyemat, dyepeps, gain_model, radmat, true_dyt_iz, true_ks, n_samples = (None,) * 7

    def _test(
        n_neighbors=4, run_against_all_dyetracks=False, run_row_k_fit=True, k_sigma=0.2
    ):
        with c_nn_v2.context(
            dyemat,
            dyepeps,
            radmat.astype(RadType),
            *gain_model,
            k_sigma=k_sigma,
            n_neighbors=n_neighbors,
            run_row_k_fit=run_row_k_fit,
            run_against_all_dyetracks=run_against_all_dyetracks,
        ) as nn_v2_context:
            c_nn_v2.do_classify_radrows(nn_v2_context, 0, len(radmat))
            return nn_v2_context

    def _before():
        nonlocal dyemat, dyepeps, gain_model, radmat, true_dyt_iz, true_ks, n_samples

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

        gain_model = (6000, 0.20, 0.0, 200.0)
        n_samples = 5
        radmat, true_dyt_iz, true_ks = _radmat_from_dyemat(
            dyemat, gain_model, n_samples=n_samples, k_sigma=0.0
        )

    def it_catches_non_sequential_dyt_iz_in_dyepeps():
        nonlocal dyepeps
        dyepeps = np.array([[1, 1, 10], [2, 1, 10], [1, 2, 30],], dtype=np.uint64,)
        with zest.raises(c_nn_v2.NNV2Exception, in_args="Non sequential dyt_i"):
            _test()

    def it_enforces_reverse_sort_on_count_per_dyt():
        nonlocal dyepeps
        dyepeps = np.array([[1, 1, 10], [1, 2, 30],], dtype=np.uint64,)
        with zest.raises(
            c_nn_v2.NNV2Exception, in_args="must be reverse sorted by count per dyt"
        ):
            _test()

    # TODO: Add a retry on assert because it can be wrong
    def it_classifies():
        nn_v2_context = _test()
        assert np.all(true_dyt_iz == nn_v2_context.pred_dyt_iz)

    def it_fits_k():
        nonlocal radmat, true_dyt_iz, true_ks
        radmat, true_dyt_iz, true_ks = _radmat_from_dyemat(
            dyemat, gain_model, n_samples=500, k_sigma=0.2
        )
        mask = true_dyt_iz > 0
        radmat = radmat[mask]
        true_dyt_iz = true_dyt_iz[mask]
        true_ks = true_ks[mask]
        nn_v2_context = _test()

        # Check that there's a reasonable correlation between true and pred k
        # I ran this several times and found with random true_ks
        # ie: true_ks = np.random.normal(1.0, 0.5, true_ks.shape[0])
        # that random true_ks generated correlations of: -0.0004, -0.001, -0.002 off diagonal
        # and that when there was a correlation it gave value like: 0.05, 0.04, 0.04
        cov = np.cov(true_ks, nn_v2_context.pred_ks)
        assert cov[1, 1] > 0.03

    def it_compares_to_all_dyetracks_without_row_fit():
        nn_v2_context = _test(
            n_neighbors=0, run_against_all_dyetracks=True, run_row_k_fit=False
        )

        # In this mode I expect to get back outputs for every radrow vs every dytrow

        assert np.all(true_dyt_iz == nn_v2_context.pred_dyt_iz)
        assert np.all(nn_v2_context.against_all_dyetrack_pred_ks == 1.0)
        assert np.all(nn_v2_context.against_all_dyetrack_pred_ks == 1.0)
        assert nn_v2_context.against_all_dyetrack_pvals.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )
        assert nn_v2_context.against_all_dyetrack_pred_ks.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )

    def it_compares_to_all_dyetracks_with_row_fit():
        nonlocal radmat, true_dyt_iz, true_ks

        k_sigma = 0.2
        radmat, true_dyt_iz, true_ks = _radmat_from_dyemat(
            dyemat, gain_model, n_samples=500, k_sigma=k_sigma
        )

        mask = true_dyt_iz > 0
        radmat = radmat[mask]
        true_dyt_iz = true_dyt_iz[mask]
        true_ks = true_ks[mask]
        nn_v2_context = _test(
            n_neighbors=0,
            run_against_all_dyetracks=True,
            run_row_k_fit=True,
            k_sigma=k_sigma,
        )

        # In this mode I expect to get back outputs for every radrow vs every dytrow
        # np.save("true_ks.npy", true_ks)
        # np.save("pred_ks.npy", nn_v2_context.pred_ks)

        assert nn_v2_context.against_all_dyetrack_pvals.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )
        assert nn_v2_context.against_all_dyetrack_pred_ks.shape == (
            radmat.shape[0],
            dyemat.shape[0],
        )

        # Check that there's a reasonable correlation between true and pred k
        # I ran this several times and found with random true_ks
        # ie: true_ks = np.random.normal(1.0, 0.5, true_ks.shape[0])
        # that random true_ks generated correlations of: -0.0004, -0.001, -0.002 off diagonal
        # and that when there was a correlation it gave value like: 0.05, 0.04, 0.04
        cov = np.cov(true_ks, nn_v2_context.pred_ks)
        assert cov[1, 1] > 0.03

    zest()
