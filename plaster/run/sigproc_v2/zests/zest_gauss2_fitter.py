import pandas as pd
import numpy as np
from plaster.run.sigproc_v2.c_gauss2_fitter import gauss2_fitter
from plaster.tools.image import imops
from zest import zest
from plaster.tools.log.log import debug

N_FIT_PARAMS = gauss2_fitter.AugmentedGauss2Params.N_PARAMS
N_FULL_PARAMS = gauss2_fitter.AugmentedGauss2Params.N_FULL_PARAMS
RHO = gauss2_fitter.AugmentedGauss2Params.RHO
OFFSET = gauss2_fitter.AugmentedGauss2Params.OFFSET


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
        for amp in np.linspace(1000.0, 500.0, 10):
            true_params, pred_params, std_params = _test(
                (amp, 1.8, 1.8, 5.0, 5.0, 0.0, 100.0),
                (amp, 1.8, 1.8, 5.0, 5.0, 0.0, 100.0),
                noise=1.5,
            )
            assert _params_close(true_params, pred_params, rel_tol=0.20)

    def it_skips_nan_locs():
        true_params = (1000, 1.8, 1.8, 5.0, 5.0, 0.0, 0.0)
        im = imops.gauss2_rho_form(*true_params, mea=11)
        locs = np.array([[5.0, 5.0], [np.nan, np.nan], [5.0, 5.0]])
        start_params = np.repeat(_full_params(true_params)[None, :], (3,), axis=0)
        pred_params, std_params = gauss2_fitter.fit_image(im, locs, start_params, 11)
        assert np.all(~np.isnan(pred_params[0, :]))
        assert np.all(np.isnan(pred_params[1, :]))
        assert np.all(~np.isnan(pred_params[2, :]))

    def fits_without_amplitude():
        # fmt: off

        # Some real data from the scope where we don't know the amplitude
        im = np.array([
            45.182922, 74.141846, 239.099731, 668.081299, 191.036194, 544.050903, 252.486694, -20.725830, 431.643066, 91.437439, 181.588806,
            32.732178, 228.824402, 79.144958, 810.321777, 315.864624, 407.177856, 151.267273, 379.886597, 375.461060, 237.137024, 113.771973,
            188.563965, 291.359558, 785.109741, 704.666138, 1130.613037, 758.278564, 820.866455, 627.234497, 304.234131, 317.114502, 84.019287,
            351.399170, 479.631592, 821.305542, 1015.433838, 1281.175049, 1834.395020, 1371.971191, 1032.910156, 536.669922, 283.952087, 69.264038,
            126.800110, 380.765503, 935.976685, 1334.455811, 2569.933838, 3465.306885, 2887.593750, 1600.522949, 490.858398, 524.587646, 46.409546,
            168.568359, 364.921387, 759.864746, 2140.803711, 3655.072998, 5829.345459, 3782.699463, 1839.223145, 240.492065, 189.823242, 162.252625,
            530.007568, 240.568359, 1084.687622, 1246.724121, 2829.401855, 4010.714111, 2635.571533, 1308.410645, 741.281006, 400.520630, -53.691895,
            207.148193, 328.579346, 296.473145, 933.114136, 958.879150, 2047.724365, 1325.949463, 1283.596069, 482.741211, 126.796570, 78.386169,
            87.321228, 341.570435, 63.425110, 357.057861, 669.304687, 926.200928, 569.316528, 385.730469, 302.688232, 289.019226, 133.026062,
            112.946106, 222.877625, 138.745178, 456.749756, 584.163208, 436.457764, 636.773315, 396.476440, 125.994751, 229.165283, 193.487976,
            -55.492004, -61.113708, 142.449158, 155.362976, 229.562866, 217.055481, 433.583008, 444.253174, 53.348145, 83.615601, 71.057312,
        ])
        # fmt: on

        im = im.reshape((11, 11))

        presumably_correct = np.array(
            [
                5.42885069e04,
                1.35490057e00,
                1.36178121e00,
                4.99880771e00,
                5.01192065e00,
                1.16176143e-02,
                2.68984531e02,
                1.10000000e01,
                2.48287853e03,
                0.00000000e00,
            ]
        )

        def it_fits_problem_1_with_reasonable_guess():
            """From a problem encountered during debugging"""
            fit_params, _ = gauss2_fitter.fit_image(
                im,
                np.array([[5, 5]]),
                np.array([[50_000, 1.4, 1.4, 5, 5, 0, 0, 0, 0, 0]]),
                11,
            )
            diff = presumably_correct - fit_params
            assert np.all(np.abs(diff) < 5e-4)

        def it_fits_when_amp_guess_is_zero():
            """
            Zero is a special case of fit_image in that it asks
            that fit_image guesses at the amplitude parameter itself
            """

            fit_params, _ = gauss2_fitter.fit_image(
                im,
                np.array([[5, 5]]),
                np.array([[0.0, 1.4, 1.4, 5, 5, 0, 0, 0, 0, 0]]),
                11,
            )
            diff = presumably_correct - fit_params
            assert np.all(np.abs(diff) < 5e-4)

        zest()

    zest()
