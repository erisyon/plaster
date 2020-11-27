from zest import zest
from plaster.tools.image import imops
from plaster.tools.image.coord import XY, YX
from plaster.run.sigproc_v2.reg_psf import RegPSF
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.c_radiometry.radiometry import radiometry_field_stack
import numpy as np
from plaster.tools.log.log import debug


def zest_radiometry():
    reg_psf = RegPSF.fixture()

    def _peak(x, y):
        return imops.gauss2_rho_form(1000.0, 1.8, 1.8, 11/2 + x, 11/2 + y, 0.0, 0.0, 11)

    def it_uses_appropriate_regional_psf():
        raise NotImplementedError

    def it_finds_one_peak_no_noise():
        n_cycles = 1
        y = 0.0
        x = 0.0
        chcy_ims = np.zeros((1, n_cycles, 11, 11))
        imops.accum_inplace(chcy_ims[0, 0], _peak(y, x), loc=XY(0, 0), center=False)
        radrow = radiometry_field_stack(chcy_ims, locs=np.array([[y, x]]) + 5.5, reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles)))
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 1e-9
        assert np.abs(radrow[0, 0, 0, 2]) > 1e9

    def it_finds_fractional_loc():
        n_cycles = 1
        for y in np.linspace(-1, 1, 5):
            for x in np.linspace(-1, 1, 5):
                chcy_ims = np.zeros((1, n_cycles, 11, 11))
                imops.accum_inplace(chcy_ims[0, 0], _peak(y, x), loc=XY(0, 0), center=False)
                radrow = radiometry_field_stack(chcy_ims, locs=np.array([[y, x]]) + 5.5, reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles)))
                assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
                assert np.abs(radrow[0, 0, 0, 1]) < 1e-9
                assert np.abs(radrow[0, 0, 0, 2]) > 1e9

    def it_adjusts_for_focus():
        raise NotImplementedError

    def it_returns_signal():
        raise NotImplementedError

    def it_returns_noise():
        raise NotImplementedError

    def it_returns_snr():
        raise NotImplementedError

    def it_returns_asr():
        raise NotImplementedError

    zest()

