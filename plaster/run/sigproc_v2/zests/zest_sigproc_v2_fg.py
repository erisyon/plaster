import numpy as np
from plaster.run.sigproc_v2 import fg
from plaster.run.sigproc_v2 import synth
from plaster.tools.log.log import debug
from zest import zest


def zest_fit_method():
    def it_agrees_kernel_and_fit_methods():
        n_peaks = 50
        amp = 1000
        dim = (512, 512)
        peak_width = 1.84
        peak_width_variance_scale = 0.30
        bg_mean = 100
        bg_std = 10
        halo_size = 20
        halo_intensity = 3
        with synth.Synth(overwrite=True, dim=dim) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=n_peaks)
                .amps_constant(val=amp)
                .widths_variable(peak_width, peak_width_variance_scale)
                .locs_randomize_away_from_edges(dist=15)
            )
            synth.CameraModel(bias=bg_mean, std=bg_std)
            synth.HaloModel(halo_size, halo_intensity)
            im = s.render_chcy()[0, 0]

        divs = 5
        psf_params = np.broadcast_to(
            np.array(
                [100.0, 1.8, 1.8, peaks.mea / 2, peaks.mea / 2, 0.0, 0.0, peaks.mea]
            ),
            (1, divs, divs, 8),
        )

        fit_params = fg.radiometry_one_channel_one_cycle_fit_method(
            im, psf_params, peaks.locs
        )
        med = np.median(fit_params[:, 0])
        if not (900 < med < 1100):
            debug(med)
        assert 900 < med < 1100

    zest()
