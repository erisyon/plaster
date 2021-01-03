import numpy as np
from plaster.run.sigproc_v2 import synth, psf, bg
from plaster.run.calib.calib import RegPSF
from plaster.run.sigproc_v2.c_radiometry.radiometry import radiometry_field_stack
from zest import zest
from plaster.tools.log.log import debug


def zest_bg():
    def it_removes_bg_by_low_cut_one_cycle():
        reg_psf = RegPSF.fixture()
        with synth.Synth(overwrite=True, dim=(512, 512), n_cycles=1) as s:
            peaks = (
                synth.PeaksModelPSF(reg_psf, n_peaks=100)
                .amps_constant(5000.0)
                .locs_grid(pad=50)
            )
            synth.CameraModel(bg_mean=100, bg_std=10)
            synth.HaloModel()
            chcy_ims = s.render_chcy()

        im, std = bg.bandpass_filter(
            chcy_ims[0, 0],
            low_inflection=0.03,
            low_sharpness=50.0,
            high_inflection=0.5,
            high_sharpness=50.0,
        )

        # CHECK that it removed the bg
        assert -2.0 < np.median(im) < 2.0

        # CHECK that it did not affect the fg by much
        bg_removed_chcy_ims = im[None, None, :, :]
        bg_removed_chcy_ims = np.ascontiguousarray(
            bg_removed_chcy_ims, dtype=np.float64
        )

        radmat = radiometry_field_stack(
            bg_removed_chcy_ims,
            locs=peaks.locs,
            reg_psf=reg_psf,
            focus_adjustment=np.ones((s.n_cycles)),
        )

    def it_removes_bg_by_low_cut():
        with synth.Synth(overwrite=True, dim=(512, 512), n_cycles=3) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .widths_uniform(1.5)
                .locs_grid(pad=50)
            )
            synth.CameraModel(bg_mean=100, bg_std=10)
            synth.HaloModel()
            peaks.dyt_random_choice(
                [[1, 1, 1], [1, 0, 1]], [0.5, 0.5]
            ).gain_constant(5000)
            chcy_ims = s.render_chcy()

        im, std = bg.bandpass_filter(
            chcy_ims[0, 0],
            low_inflection=0.03,
            low_sharpness=50.0,
            high_inflection=0.5,
            high_sharpness=50.0,
        )
        median, high = np.percentile(im, (50, 99.9))
        assert np.abs(median) < 1.0
        assert np.abs(high) < 400.0

    zest()
