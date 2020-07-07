from zest import zest
from plaster.run.sigproc_v1.zests.synth import Synth
from plaster.tools.image.coord import XY, YX, WH, HW, ROI

# TASK: Expand these tests, but this module is already well tested by the peak_finder tests


def zest_synth():
    def it_builds_multi_channel():
        synth = Synth(
            n_peaks=10,
            n_cycles=3,
            n_channels=2,
            dim=WH(100, 100),
            peak_mean=(60, 120),
            bg_std=1.0,
        )
        assert synth.ims.shape == (2, 3, 100, 100)
        assert (
            synth.frame_xs[0] == 0 and synth.frame_ys[0] == 0
        )  # The offset of the zero-th frame should be zero
        assert synth.n_peaks == 10
        assert synth.n_cycles == 3
        assert synth.n_channels == 2

    zest()
