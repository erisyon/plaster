import numpy as np
from plaster.run.sigproc_v2 import fg
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.c.gauss2_fitter import AugmentedGauss2Params
from plaster.tools.log.log import debug
from plaster.tools.utils import utils
from zest import zest


def zest_fit_method():
    mea = 11
    bg_mean = 10

    def _run(amp=1000, peak_width=1.84, peak_height=1.84, peak_shift_x=0.0, n_peaks=50):
        dim = (512, 512)
        bg_std = 5
        halo_size = 20
        halo_intensity = 3

        with synth.Synth(overwrite=True, dim=dim) as s:
            peaks = (
                synth.PeaksModelGaussian(n_peaks=n_peaks, mea=mea)
                .amps_constant(val=amp)
                .uniform_width_and_heights(peak_width, peak_height)
                .locs_randomize_away_from_edges(dist=15)
            )
            if peak_shift_x > 0.0:
                peaks.locs = np.floor(peaks.locs)
                peaks.locs[:, 1] += peak_shift_x
            synth.CameraModel(bias=bg_mean, std=bg_std)
            # synth.HaloModel(halo_size, halo_intensity)
            im = s.render_chcy()[0, 0]

        divs = 5
        psf_params = np.broadcast_to(
            np.array(
                [
                    amp,
                    peak_height,
                    peak_width,
                    peaks.mea / 2,
                    peaks.mea / 2,
                    0.0,
                    0.0,
                    peaks.mea,
                ]
            ),
            (1, divs, divs, 8),
        )

        fit_params = fg.radiometry_one_channel_one_cycle_fit_method(
            im, psf_params, peaks.locs
        )

        return fit_params

    def it_fits():
        fit_params = _run()

        med = np.nanmedian(fit_params[:, AugmentedGauss2Params.SIGNAL])
        assert 900 < med < 1100

        h = np.nanmedian(fit_params[:, AugmentedGauss2Params.SIGMA_Y])
        assert 1.7 < h < 1.9

        w = np.nanmedian(fit_params[:, AugmentedGauss2Params.SIGMA_X])
        assert 1.7 < w < 2.0

        cy = np.nanmedian(fit_params[:, AugmentedGauss2Params.CENTER_Y])
        assert 5.0 < cy < 6.0

        cx = np.nanmedian(fit_params[:, AugmentedGauss2Params.CENTER_X])
        assert 5.0 < cx < 6.0

        rho = np.nanmedian(fit_params[:, AugmentedGauss2Params.RHO])
        assert -0.05 < rho < 0.05

        off = np.nanmedian(fit_params[:, AugmentedGauss2Params.OFFSET])
        assert bg_mean * 0.5 < off < bg_mean * 1.5

        assert np.all(
            (fit_params[:, AugmentedGauss2Params.MEA] == mea)
            | np.isnan(fit_params[:, AugmentedGauss2Params.MEA])
        )

    def it_handles_all_noise():
        fit_params = _run(amp=0)
        n_nan = np.isnan(fit_params[:, 0]).sum()
        assert n_nan > fit_params.shape[0] - 5

    def it_handles_wide_peaks():
        fit_params = _run(peak_width=2.7, peak_height=1.5)
        w = np.nanmedian(fit_params[:, AugmentedGauss2Params.SIGMA_X])
        h = np.nanmedian(fit_params[:, AugmentedGauss2Params.SIGMA_Y])
        assert 2.55 < w < 2.8
        assert 1.4 < h < 1.6

    def it_handles_tall_peaks():
        fit_params = _run(peak_width=1.5, peak_height=2.7)
        w = np.nanmedian(fit_params[:, AugmentedGauss2Params.SIGMA_X])
        h = np.nanmedian(fit_params[:, AugmentedGauss2Params.SIGMA_Y])
        assert 2.55 < h < 2.8
        assert 1.4 < w < 1.6

    def it_handles_shifted_right_peaks():
        fit_params = _run(peak_shift_x=0.2)
        x = np.nanmedian(fit_params[:, AugmentedGauss2Params.CENTER_X])
        y = np.nanmedian(fit_params[:, AugmentedGauss2Params.CENTER_Y])
        assert 5.15 < x < 5.25
        assert 4.95 < y < 5.15

    def it_handles_collisions():
        fit_params = _run(n_peaks=50)
        n_nans = np.isnan(fit_params[:, 0]).sum()
        assert n_nans < 10

        fit_params = _run(n_peaks=500)
        n_nans = np.isnan(fit_params[:, 0]).sum()
        assert n_nans < 220

    zest()
