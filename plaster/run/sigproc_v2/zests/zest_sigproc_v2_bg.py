import numpy as np
from plaster.run.sigproc_v2 import synth, psf, bg
from zest import zest


def zest_bg():
    def it_removes_bg_by_low_cut():
        with synth.Synth(overwrite=True, dim=(512, 512), n_cycles=3) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .widths_uniform(1.5)
                .locs_randomize()
            )
            synth.CameraModel(100, 100)
            synth.HaloModel()
            peaks.dyt_amp_constant(5000).dyt_random_choice(
                [[1, 1, 1], [1, 0, 1]], [0.5, 0.5]
            )
            chcy_ims = s.render_chcy()

        kernel = psf.approximate_psf()
        im, mean, std = bg.bg_remove(chcy_ims[0, 0], kernel)
        median, high = np.percentile(im, (50, 99.9))

        assert np.abs(median) < 1.0
        assert np.abs(high) < 3.8

    zest()
