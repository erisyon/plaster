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

    def _peak(x, y, mea):
        return imops.gauss2_rho_form(1000.0, 1.8, 1.8, x, y, 0.0, 0.0, mea)

    def it_finds_one_peak_centered():
        n_cycles = 1
        cen = 11/2
        y = 0.0 + cen
        x = 0.0 + cen
        chcy_ims = np.zeros((1, n_cycles, 11, 11))
        imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 11), loc=XY(0, 0), center=False)
        radrow = radiometry_field_stack(chcy_ims, locs=np.array([[y, x]]), reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles)))
        # print(f"{y:8.2f} {x:8.2f} {radrow[0, 0, 0, 0]:8.2f} {radrow[0, 0, 0, 1]:8.2f}")
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001
        assert np.abs(radrow[0, 0, 0, 2]) > 1e6

    def it_finds_off_center():
        n_cycles = 1
        cen = 21/2
        y = -1.0 + cen
        x = 1.0 + cen
        chcy_ims = np.zeros((1, n_cycles, 21, 21))
        imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 21), loc=XY(0,0), center=False)
        radrow = radiometry_field_stack(chcy_ims, locs=np.array([[y, x]]), reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles)))
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001

    def it_finds_way_off_center():
        n_cycles = 1
        cen = 21/2
        y = -2.2 + cen
        x = 5.7 + cen
        chcy_ims = np.zeros((1, n_cycles, 21, 21))
        imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 21), loc=XY(0,0), center=False)
        # np.save("./_test.npy", chcy_ims[0, 0])
        radrow = radiometry_field_stack(chcy_ims, locs=np.array([[y, x]]), reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles)))
        # print(f"{y:8.2f} {x:8.2f} {radrow[0, 0, 0, 0]:8.2f} {radrow[0, 0, 0, 1]:8.2f}")
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001

    def it_finds_fractional_loc():
        n_cycles = 1
        cen = 21/2
        for y in np.linspace(cen-2, cen+2, 14):
            for x in np.linspace(cen-2, cen+2, 14):
                chcy_ims = np.zeros((1, n_cycles, 21, 21))
                imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 21), loc=XY(0,0), center=False)
                radrow = radiometry_field_stack(chcy_ims, locs=np.array([[y, x]]), reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles)))
                # print(f"{y:8.2f} {x:8.2f} {radrow[0,0,0,0]:8.2f} {radrow[0,0,0,1]:8.2f}")
                assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
                assert np.abs(radrow[0, 0, 0, 1]) < 0.001

    # def it_uses_appropriate_regional_psf():
    #     raise NotImplementedError
    #
    # def it_adjusts_for_focus():
    #     raise NotImplementedError
    #
    # def it_returns_signal():
    #     raise NotImplementedError
    #
    # def it_returns_noise():
    #     raise NotImplementedError
    #
    # def it_returns_snr():
    #     raise NotImplementedError
    #
    # def it_returns_asr():
    #     raise NotImplementedError

    zest()

