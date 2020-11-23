from zest import zest
from plaster.tools.image import imops
from plaster.run.sigproc_v2 import synth
import numpy as np
from plaster.tools.log.log import debug


@zest.skip(reason="Need a massive overhaul since refactor")
def zest_align():
    def _ims(mea=512, std=1.5):
        bg_mean = 145
        with synth.Synth(n_cycles=3, overwrite=True, dim=(mea, mea)) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=1000)
                .amps_constant(val=4_000)
                .locs_randomize()
                .widths_uniform(std)
            )
            synth.CameraModel(bias=bg_mean, std=14)
            cy_ims = s.render_chcy()[0]
            return cy_ims, s.aln_offsets

    def it_removes_the_noise_floor():
        cy_ims, true_aln_offsets = _ims()
        pred_aln_offsets, aln_scores = worker._analyze_step_3_align(cy_ims)
        assert np.all(true_aln_offsets == pred_aln_offsets)

    def it_is_robust_to_different_image_sizes():
        cy_ims, true_aln_offsets = _ims(mea=128)
        pred_aln_offsets, aln_scores = worker._analyze_step_3_align(cy_ims)
        assert np.all(true_aln_offsets == pred_aln_offsets)

    def it_is_robust_to_different_peak_sizes():
        cy_ims, true_aln_offsets = _ims(std=3.0)
        pred_aln_offsets, aln_scores = worker._analyze_step_3_align(cy_ims)
        assert np.all(true_aln_offsets == pred_aln_offsets)

    zest()


@zest.skip(reason="WIP")
def zest_sub_pixel_align():
    def _synth_cycles():
        n_peaks = 500
        amp = 5000
        dim = (512, 512)

        bg_mean = 0
        bg_std = 100
        n_cycles = 3
        with synth.Synth(overwrite=True, dim=dim, n_cycles=n_cycles) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=n_peaks)
                .amps_constant(val=amp)
                .widths_uniform(width=1.8)
                .locs_randomize()
            )
            synth.CameraModel(bias=bg_mean, std=bg_std)
            s.zero_aln_offsets()
            cy_ims = s.render_chcy()[0, :]

        return cy_ims, peaks.locs

    def it_aligns():
        import pudb; pudb.set_trace()
        cy_ims, locs = _synth_cycles()

        # Simulate sub-pixel shift
        true_offsets = np.array([[0.0, 0.0], [1.235, 3.728], [-3.492, 8.911],])

        im0 = cy_ims[0]
        im1 = imops.sub_pixel_shift(cy_ims[1], true_offsets[1])
        im2 = imops.sub_pixel_shift(cy_ims[2], true_offsets[2])
        im_stack = np.stack((im0, im1, im2)).astype(np.float32)
        pred_offsets = imops.sub_pixel_align(im_stack, n_divs=2, precision=10)
        diff = pred_offsets - true_offsets
        assert np.all(np.abs(diff) <= 0.1)

    zest()
