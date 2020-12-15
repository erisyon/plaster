import numpy as np

import plaster.run.calib.calib
import plaster.run.sigproc_v2.psf
from plaster.run.sigproc_v2 import synth, bg, fg, psf
from plaster.run.calib.calib import approximate_psf
from plaster.tools.image.coord import HW
from plaster.tools.image import imops
from plaster.tools.log.log import debug
from zest import zest


def zest_peak_find():
    def it_finds_one_peak_sub_pixel_exactly_under_ideal_conditions():
        true_locs = np.array([[17.5, 17.5]])
        peak_im = imops.gauss2_rho_form(
            amp=1000.0,
            std_x=1.8,
            std_y=1.8,
            pos_x=true_locs[0, 1],
            pos_y=true_locs[0, 0],
            rho=0.0,
            const=0.0,
            mea=35,
        )

        pred_locs = fg._sub_pixel_peak_find(peak_im, HW(11, 11), true_locs.astype(int))
        dists = np.linalg.norm(true_locs - pred_locs, axis=1)
        assert np.all(dists < 0.001)

    def it_finds_one_peak_sub_pixel_exactly_under_ideal_conditions_many_offsets():
        for trials in range(50):
            true_locs = np.random.uniform(-5, 5, size=(1, 2))
            true_locs += 35 / 2
            peak_im = imops.gauss2_rho_form(
                amp=1000.0,
                std_x=1.8,
                std_y=1.8,
                pos_x=true_locs[0, 1],
                pos_y=true_locs[0, 0],
                rho=0.0,
                const=0.0,
                mea=35,
            )

            pred_locs = fg._sub_pixel_peak_find(
                peak_im, HW(11, 11), true_locs.astype(int)
            )
            dists = np.linalg.norm(true_locs - pred_locs, axis=1)
            assert np.all(dists < 0.001)

    def it_find_pixel_accurate():
        bg_std = 10
        with synth.Synth(overwrite=True, dim=(512, 512), n_cycles=3) as s:
            true_n_peaks = 100
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=true_n_peaks)
                .amps_constant(1000)
                .widths_uniform(1.5)
                .locs_randomize()
            )
            s.zero_aln_offsets()
            synth.CameraModel(0, bg_std)
            chcy_ims = s.render_chcy()

        im, bg_std = bg.bandpass_filter(
            chcy_ims[0, 0],
            low_inflection=0.03,
            low_sharpness=50.0,
            high_inflection=0.5,
            high_sharpness=50.0,
        )
        kernel = plaster.run.calib.calib.approximate_psf()
        locs = fg.peak_find(chcy_ims[0, 0], kernel, np.mean(bg_std))
        n_peaks, n_dims = locs.shape
        assert n_dims == 2
        assert n_peaks > 0.85 * true_n_peaks

    def it_finds_sub_pixel_exactly_under_ideal_conditions():
        """
        Test the helper _sub_pixel_peak_find instead of sub_pixel_peak_find
        because we don't want to have to reconcile the peak ordering
        from the synth with the arbitrary order they are found by the
        peak finder
        """

        with synth.Synth(overwrite=True, dim=(512, 512), n_cycles=3) as s:
            true_n_peaks = 100
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=true_n_peaks)
                .amps_constant(1000)
                .widths_uniform(1.5)
                .locs_grid()
                .locs_add_random_subpixel()
            )
            s.zero_aln_offsets()
            chcy_ims = s.render_chcy()

        chcy_mean_im = np.mean(chcy_ims, axis=(0, 1))
        locs = fg._sub_pixel_peak_find(
            chcy_mean_im, HW(peaks.mea, peaks.mea), peaks.locs.astype(int)
        )
        dists = np.linalg.norm(locs - peaks.locs, axis=1)
        assert np.all(dists < 0.01)

    def it_finds_sub_pixel_well_under_typical_conditions():
        bg_std = 10
        with synth.Synth(overwrite=True, dim=(512, 512), n_cycles=3) as s:
            true_n_peaks = 100
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=true_n_peaks)
                .amps_constant(1000)
                .widths_uniform(1.5)
                .locs_randomize_away_from_edges()
            )
            synth.CameraModel(0, bg_std)
            s.zero_aln_offsets()
            chcy_ims = s.render_chcy()

        chcy_mean_im = np.mean(chcy_ims, axis=(0, 1))
        locs = fg._sub_pixel_peak_find(
            chcy_mean_im, HW(peaks.mea, peaks.mea), peaks.locs.astype(int)
        )
        dists = np.linalg.norm(locs - peaks.locs, axis=1)

        assert (dists < 0.1).sum() > 30
        assert (dists < 0.2).sum() > 70

    zest()
