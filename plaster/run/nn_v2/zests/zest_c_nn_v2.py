import numpy as np
from plaster.run.sim_v2.sim_v2_result import RadType, DyeType
from plaster.run.nn_v2.c import nn_v2 as c_nn_v2
from scipy.stats import norm
from zest import zest
from plaster.tools.log.log import debug


def sample_gaussian(beta, sigma, n_samples):
    return norm(beta, sigma).rvs(n_samples)


def _radmat_from_dyemat(dyemat, gain_model, n_samples):
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
    return radmat, true_dyt_iz


def zest_c_nn_v2():
    dyemat, dyepeps, gain_model, radmat, true_dyt_iz = (None,) * 5

    def _test():
        with c_nn_v2.context(
            dyemat,
            dyepeps,
            radmat.astype(RadType),
            *gain_model,
            row_k_std=0.2,
            n_neighbors=4,
            run_row_k_fit=True,
            run_against_all_dyetracks=False,
        ) as nn_v2_context:
            c_nn_v2.do_classify_radrows(nn_v2_context, 0, len(radmat))
            return nn_v2_context

    def _before():
        nonlocal dyemat, dyepeps, gain_model, radmat, true_dyt_iz

        # fmt: off
        dyemat = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 1, 1],
            # [3, 2, 1],
        ], dtype=DyeType)
        # fmt: on

        dyepeps = np.array(
            [
                # (dyt_i, pep_i, count)
                [1, 2, 30],
                # [1, 1, 10],
                [2, 1, 10],
                # [3, 1, 10],
            ],
            dtype=np.uint64,
        )

        gain_model = (6000, 0.20, 0.0, 200.0)

        radmat, true_dyt_iz = _radmat_from_dyemat(dyemat, gain_model, n_samples=1)  # HACK FIX ME

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

    def it_classifies():
        nn_v2_context = _test()
        debug(true_dyt_iz)
        debug(nn_v2_context.pred_dyt_iz)
        assert np.all(true_dyt_iz == nn_v2_context.pred_dyt_iz)

    zest()
