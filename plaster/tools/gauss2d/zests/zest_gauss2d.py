from zest import zest
import numpy as np
from plaster.tools.image import imops
from plaster.tools.gauss2d import gauss2d
from plaster.run.sigproc_v2 import synth
from plaster.tools.log.log import debug


def _dump_params(params):
    print("".join([f"{p:3.4f} " for p in params]))


def zest_run_gauss_2_fit_on_synthetic():
    raise NotImplementedError

    # TODO More to do here, plus I'd like to get a timing

    lib = gauss2d.load_lib()

    with synth.Synth() as s:
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=100)
            .locs_grid()
            .widths_uniform(1.8)
            .amps_constant(val=6000)
        )
        synth.CameraModel(bias=0, std=10)
        im = s.render_chcy()[0, 0].astype(np.float32)

    # np.save("test.npy", im)

    locs = np.array(peaks.locs)
    cen_y = locs[:, 0].astype(int)
    cen_x = locs[:, 1].astype(int)

    mea = 9
    y = int(cen_y[0])
    x = int(cen_x[0])
    _im = im[y - 4 : y + 5, x - 4 : x + 5]

    n_peaks = peaks.n_peaks
    n_peaks = 1  # HACK
    params = np.zeros(
        (n_peaks, 7)
    )  # (amplitude, sigma_x, sigma_y, pos_x, pos_y, rho, offset)
    fails = np.zeros((n_peaks,), dtype=int)

    params[:, 0] = 1010.0
    params[:, 1] = peaks.std_x[0:n_peaks]
    params[:, 2] = peaks.std_y[0:n_peaks]
    params[:, 3] = mea / 2
    params[:, 4] = mea / 2
    params[:, 5] = 0.0
    params[:, 6] = 0.0

    params = params.flatten()

    n_fails = lib.fit_array_of_gauss_2d_on_float_image(
        im, im.shape[0], im.shape[1], mea, n_peaks, cen_y, cen_x, params, fails,
    )

    print(f"n_peaks={n_peaks}")
    print(f"n_fails={n_fails}")
    print(params[0:7])
    print(fails[0])

    zest()


def zest_gauss_2_fit():
    lib = gauss2d.load_lib()

    mea = 9

    def _peak():
        true_params = (
            1000.0,  # amp
            1.8,  # sig_y
            1.2,  # sig_x
            5.0,  # pos_y
            4.8,  # pos_x
            0.05,  # rho
            50.0,  # offset
        )

        im = imops.gauss2_rho_form(
            true_params[0],
            true_params[2],  # Note reversed index
            true_params[1],
            true_params[4],  # Note reversed index
            true_params[3],
            true_params[5],
            true_params[6],
            mea,
        ).flatten()

        im = im + np.random.normal(0, 0.1, im.shape)

        # Perturb the parameters
        fit_params = np.array(
            [
                1010.0,  # amp
                1.9,  # sig_y
                1.0,  # sig_x
                5.1,  # pos_y
                4.1,  # pos_x
                0.00,  # rho
                0.0,  # offset
            ]
        )

        return im, true_params, fit_params

    def it_fits_a_single_gauss_2d():
        im, true_params, fit_params = _peak()
        info = np.zeros((10,))
        covar = np.zeros((7 * 7,))
        failed = lib.fit_gauss_2d(im, mea, fit_params, info, covar)
        assert failed == 0
        assert np.allclose(fit_params, true_params, rtol=0.06)

    def it_fits_one_peak_on_float_image():
        im, true_params, fit_params = _peak()

        info = np.zeros((10,))
        covar = np.zeros((7 * 7,))
        im = im.astype(np.float32).reshape((mea, mea))
        bigger_im = np.zeros((512, 512), dtype=np.float32)
        bigger_im[100 : 100 + mea, 100 : 100 + mea] = im
        failed = lib.fit_gauss_2d_on_float_image(
            bigger_im,
            bigger_im.shape[0],
            bigger_im.shape[1],
            100 + mea // 2,
            100 + mea // 2,
            mea,
            fit_params,
            info,
            covar,
        )
        assert failed == 0
        assert np.allclose(fit_params, true_params, rtol=0.06)

    def it_fits_array_of_gauss_2d_on_float_image():
        im, true_params, fit_params = _peak()

        im = im.astype(np.float32).reshape((mea, mea))

        bigger_im = np.zeros((512, 512), dtype=np.float32)
        bigger_im[100 : 100 + mea, 100 : 100 + mea] = im
        bigger_im[200 : 200 + mea, 200 : 200 + mea] = im

        cen_y = np.array([100, 200, 300], dtype=np.int64) + mea // 2
        cen_x = np.array([100, 200, 300], dtype=np.int64) + mea // 2

        n_peaks = 3
        fit_params = np.tile(fit_params, n_peaks)

        fails = np.zeros((n_peaks,), dtype=int)

        n_fails = lib.fit_array_of_gauss_2d_on_float_image(
            bigger_im,
            bigger_im.shape[0],
            bigger_im.shape[1],
            mea,
            n_peaks,
            cen_y,
            cen_x,
            fit_params,
            fails,
        )
        assert n_fails == 1

        fit_params = fit_params.reshape((n_peaks, 7))
        fit_params[fails == 1, :] = np.nan

        assert np.allclose(fit_params[0], true_params, rtol=0.06)
        assert np.allclose(fit_params[1], true_params, rtol=0.06)
        assert np.all(np.isnan(fit_params[2]))

    zest()
