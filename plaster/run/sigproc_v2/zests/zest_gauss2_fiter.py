import pandas as pd
import numpy as np
from plaster.run.sigproc_v2.c import gauss2_fitter
from plaster.tools.image import imops
from zest import zest
from plaster.tools.log.log import debug

N_FIT_PARAMS = gauss2_fitter.Gauss2FitParams.N_FIT_PARAMS
N_FULL_PARAMS = gauss2_fitter.Gauss2FitParams.N_FULL_PARAMS
RHO = gauss2_fitter.Gauss2FitParams.RHO
OFFSET = gauss2_fitter.Gauss2FitParams.OFFSET


def zest_gauss2_fitter():
    # params are in order:
    # (amp, std_x, std_y, pos_x, pos_y, rho, const)

    def _params_close(true_params, pred_params, rel_tol=0.12):
        true_params = np.array(true_params)
        pred_params = pred_params[0, 0:N_FIT_PARAMS]
        df = pd.DataFrame(dict(true=true_params, pred=pred_params))
        df["abs_diff"] = df.pred - df.true
        df["rel_diff"] = (df.pred - df.true) / df.true
        df["close"] = df.rel_diff ** 2 < rel_tol ** 2

        if true_params[RHO] == 0.0:
            df.loc[RHO, "close"] = pred_params[RHO] < 0.20

        if true_params[OFFSET] == 0.0:
            df.loc[OFFSET, "close"] = pred_params[OFFSET] < 10.0

        all_close = np.all(df.close)
        if not all_close:
            print("ERROR, gauss2 test failed, the parameters were:")
            print(df)
        return all_close

    def _full_params(partial_params):
        assert len(partial_params) == N_FIT_PARAMS
        p = np.zeros((N_FULL_PARAMS))
        p[0:N_FIT_PARAMS] = partial_params
        return p

    def _test(true_params, start_params=None, noise=None):
        im = imops.gauss2_rho_form(*true_params, mea=11)
        locs = np.array([[5.0, 5.0]])
        if start_params is None:
            start_params = true_params
        if noise is not None:
            im = im + np.random.normal(0, noise, size=im.shape)
        # np.save("_test.npy", im)
        start_params = _full_params(start_params)
        pred_params, std_params = gauss2_fitter.fit_image(
            im, locs, np.array([start_params]), 11
        )
        return true_params, pred_params, std_params

    def it_fits_clean():
        true_params, pred_params, _ = _test((1000, 1.8, 1.8, 5.0, 5.0, 0.0, 0.0))
        assert _params_close(true_params, pred_params)

    def it_fits_large_offset():
        true_params, pred_params, _ = _test(
            (1000, 1.8, 1.8, 5.0, 5.0, 0.0, 100.0), noise=2.0
        )
        assert _params_close(true_params, pred_params)

    def it_fits_large_offset_with_bad_initial_guess():
        true_params, pred_params, _ = _test(
            (1000, 1.8, 1.8, 5.0, 5.0, 0.0, 100.0),
            (0, 1.8, 1.8, 5.0, 5.0, 0.0, 0.0),
            noise=2.0,
        )
        assert _params_close(true_params, pred_params)

    def it_measures_parameter_variance():
        """
        Analyze how fitter responds as the signal drops.
        For each parameter I need to understand how the distribution
        of the std_params is related to the error
        """
        for trial in range(10):
            for amp in np.linspace(1000.0, 500.0, 10):
                true_params, pred_params, std_params = _test(
                    (amp, 1.8, 1.8, 5.0, 5.0, 0.0, 100.0),
                    (amp, 1.8, 1.8, 5.0, 5.0, 0.0, 100.0),
                    noise=1.5,
                )
                assert _params_close(true_params, pred_params, rel_tol=0.20)

    zest()
