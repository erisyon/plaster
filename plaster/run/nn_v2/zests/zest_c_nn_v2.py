import numpy as np
from plaster.run.sim_v2.sim_v2_result import RadType, DyeType
from plaster.run.nn_v2.c import nn_v2 as c_nn_v2
from scipy.stats import norm
from zest import zest
from plaster.tools.log.log import debug


def sample_gaussian(beta, sigma, n_samples):
    return norm(beta, sigma).rvs(n_samples)


def _radmat_from_dyemat(dyemat, n_samples, gain_model):
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
    # fmt: off
    dyemat = np.array([
        [0, 0, 0],
        [3, 2, 1],
        [2, 1, 1],
        [1, 0, 0],
    ], dtype=DyeType)
    # fmt: on

    dyepeps = np.array(
        [
            # (dyt_i, pep_i, count)
            [1, 1, 10],
            [2, 1, 10],
            [3, 1, 10],
            [1, 2, 30],
        ],
        dtype=np.uint64,
    )

    gain_model = (6000, 0.20, 0.0, 200.0)
    radmat, true_dyt_iz = _radmat_from_dyemat(dyemat[1:], 5, gain_model)

    nn_v2_context = c_nn_v2.context_create(
        dyemat,
        dyepeps,
        radmat.astype(RadType),
        *gain_model,
        row_k_std=0.2,
        n_neighbors=4,
        run_row_k_fit=True,
        run_against_all_dyetracks=False,
    )

    c_nn_v2.classify_radrows(0, len(radmat), nn_v2_context)

    zest()
