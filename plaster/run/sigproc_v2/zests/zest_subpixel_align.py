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
        bg_std = 0
        n_cycles = 3
        with synth.Synth(overwrite=True, dim=dim, n_cycles=n_cycles) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=n_peaks)
                .amps_constant(val=amp)
                .widths_uniform(width=1.8)
                .locs_randomize()
            )
            synth.CameraModel(bg_mean=bg_mean, bg_std=bg_std)

            # s.zero_aln_offsets()
            s.aln_offsets[1] = (0.200, -2.900)
            s.aln_offsets[2] = (3.510, -4.870)

            cy_ims = s.render_chcy()[0, :]

        return cy_ims, s.aln_offsets

    def it_aligns_one_spot():
        for y_off in np.linspace(-2.0, 2.0, 7):
            for x_off in np.linspace(-2.0, 2.0, 7):
                cy_ims = np.zeros((2, 21, 21))

                center = YX(21 / 2, 21 / 2)

                true_aln = np.array([[0, 0], [y_off, x_off],])

                for i in range(2):
                    cy_ims[i] = imops.gauss2_rho_form(
                        1000.0,
                        2.0,
                        2.0,
                        pos_y=center.y + true_aln[i, 0],
                        pos_x=center.x + true_aln[i, 1],
                        rho=0.0,
                        const=0.0,
                        mea=21,
                    )

                pred_aln = sub_pixel_align_cy_ims(cy_ims, slice_h=5)
                diff = np.abs(pred_aln - true_aln)
                assert np.all(np.abs(diff) <= 0.05)

    def it_aligns_full_image():
        cy_ims, true_aln = _synth_cycles()
        pred_aln = sub_pixel_align_cy_ims(cy_ims, slice_h=50)
        diff = pred_aln - true_aln

        assert np.all(np.abs(diff) <= 0.05)

    zest()
