from zest import zest
from plaster.tools.image import imops
from plaster.tools.image.coord import XY
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.c_sub_pixel_align.sub_pixel_align import (
    sub_pixel_align_cy_ims,
)
import numpy as np
from plaster.tools.log.log import debug


def zest_sub_pixel_align():
    def _synth_cycles():
        n_peaks = 800
        amp = 6000
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

            # s.zero_aln_offsets()
            s.aln_offsets[1] = (0.200, -2.900)
            s.aln_offsets[2] = (3.510, -4.870)

            cy_ims = s.render_chcy()[0, :]

        return cy_ims, s.aln_offsets

    def it_aligns_one_spot():
        cy_ims = np.zeros((2, 21, 21))

        peak_im = imops.gauss2_rho_form(1000.0, 2.0, 2.0, 5.5, 5.5, 0.0, 0.0, 11)
        imops.accum_inplace(cy_ims[0], peak_im, XY(10, 10), center=True)

        peak_im = imops.gauss2_rho_form(1000.0, 2.0, 2.0, 5.8, 4.9, 0.0, 0.0, 11)
        imops.accum_inplace(cy_ims[1], peak_im, XY(10, 10), center=True)

        pred_aln = sub_pixel_align_cy_ims(cy_ims)

        diff = pred_aln - np.array([10.5, 10.5])
        assert np.all(np.abs(diff) <= 0.1)

    def it_aligns_full_image():
        cy_ims, true_aln = _synth_cycles()

        pred_aln = sub_pixel_align_cy_ims(cy_ims)

        diff = pred_aln - true_aln
        assert np.all(np.abs(diff) <= 0.1)

    zest()
