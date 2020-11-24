from zest import zest
from plaster.tools.image import imops
from plaster.tools.image.coord import XY, YX
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

        peak_im = imops.gauss2_rho_form(
            1000.0, 2.0, 2.0, pos_y=5.5, pos_x=5.5, rho=0.0, const=0.0, mea=11
        )
        imops.accum_inplace(cy_ims[0], peak_im, YX(10, 10), center=True)

        peak_im = imops.gauss2_rho_form(
            1000.0, 2.0, 2.0, pos_y=5.7, pos_x=5.6, rho=0.0, const=0.0, mea=11
        )
        imops.accum_inplace(cy_ims[1], peak_im, YX(10 + 1, 10 - 2), center=True)

        pred_aln = sub_pixel_align_cy_ims(cy_ims)

        assert np.all(pred_aln[0, :] == 0)

        diff = pred_aln[1, :] - np.array([1.0 + 0.2, -2.0 + 0.1])
        assert np.all(np.abs(diff) <= 0.01)

    def it_aligns_full_image():
        cy_ims, true_aln = _synth_cycles()

        pred_aln = sub_pixel_align_cy_ims(cy_ims)

        debug(pred_aln, true_aln)

        diff = pred_aln - true_aln
        assert np.all(np.abs(diff) <= 0.1)

    zest()
