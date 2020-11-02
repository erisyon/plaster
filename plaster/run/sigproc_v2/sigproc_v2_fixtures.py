import numpy as np
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.run.sigproc_v2 import sigproc_v2_common
from plaster.run.sim_v2.sim_v2_worker import sim_v2
from plaster.run.error_model import GainModel
from plaster.tools.log.log import debug
from plaster.tools.schema import check
from plaster.run.sim_v2.sim_v2_fixtures import result_from_prep_fixture
from scipy.stats import norm


def _sample_gaussian(beta, sigma, n_samples):
    return norm(beta, sigma).rvs(n_samples)


def synthetic_radmat_from_dyemat(dyemat, gain_model, n_samples):
    check.t(gain_model, GainModel)
    assert gain_model.n_channels == 1
    n_dyts, n_cols = dyemat.shape
    radmat = np.zeros((n_dyts * n_samples, n_cols))
    true_dyt_iz = np.zeros((n_dyts * n_samples,), dtype=int)
    for dyt_i, dyt in enumerate(dyemat):
        dyt_radmat = np.zeros((n_samples, n_cols))
        for col_i, dye_count in enumerate(dyt):
            if dye_count > 0:
                dyt_radmat[:, col_i] = np.exp(
                    _sample_gaussian(
                        np.log(gain_model.channels[0].beta * dye_count),
                        gain_model.channels[0].sigma,
                        n_samples,
                    )
                )
            else:
                dyt_radmat[:, col_i] = _sample_gaussian(
                    gain_model.channels[0].zero_beta,
                    gain_model.channels[0].zero_sigma,
                    n_samples,
                )

        radmat[dyt_i * n_samples : (dyt_i + 1) * n_samples, :] = dyt_radmat

        true_dyt_iz[dyt_i * n_samples : (dyt_i + 1) * n_samples] = dyt_i

    n_radrows = radmat.shape[0]
    true_ks = np.ones((n_radrows,))
    if gain_model.row_k_sigma > 0.0:
        true_ks = _sample_gaussian(1.0, gain_model.row_k_sigma, n_radrows)
        radmat = radmat * true_ks[:, None]

    return radmat, true_dyt_iz, true_ks


class SigprocV2ResultFixture(SigprocV2Result):
    def _load_field_prop(self, field_i, prop):
        if prop == "signal_radmat":
            sim_v2_result = result_from_prep_fixture(self.prep_result)
            radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dyemat(
                sim_v2_result.train_dyemat, GainModel.test_fixture(), n_samples=100,
            )

        else:
            raise NotImplementedError(
                f"SigprocV2ResultFixture request un-handled prop {prop}"
            )

    @property
    def n_fields(self):
        return 1


def simple_sigproc_v2_result_fixture(prep_result):
    params = SigprocV2Params(
        calibration_file=None, mode=sigproc_v2_common.SIGPROC_V2_INSTRUMENT_ANALYZE
    )
    return SigprocV2ResultFixture(
        params=params,
        n_input_channels=1,
        n_channels=1,
        n_cycles=4,
        prep_result=prep_result,
    )
