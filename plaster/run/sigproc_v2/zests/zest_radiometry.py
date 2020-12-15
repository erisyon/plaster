from zest import zest
from plaster.tools.image import imops
from plaster.tools.image.coord import XY, YX
from plaster.run.calib.calib import RegPSF
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.c_radiometry.radiometry import (
    radiometry_field_stack,
    test_interp,
)
import numpy as np
from plaster.tools.log.log import debug


def zest_radiometry():
    reg_psf = RegPSF.fixture()

    def _peak(x, y, mea, amp=1000):
        return imops.gauss2_rho_form(amp, 1.8, 1.8, x, y, 0.0, 0.0, mea)

    def it_interpolates_correctly():
        assert test_interp()

    def it_finds_one_peak_centered():
        peak_mea = reg_psf.peak_mea
        n_cycles = 1
        cen = peak_mea / 2
        y = 0.0 + cen
        x = 0.0 + cen
        chcy_ims = np.zeros((1, n_cycles, peak_mea, peak_mea))
        imops.accum_inplace(
            chcy_ims[0, 0], _peak(x, y, peak_mea), loc=XY(0, 0), center=False
        )
        radrow = radiometry_field_stack(
            chcy_ims,
            locs=np.array([[y, x]]),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((n_cycles)),
        )
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001

        # Since there is no noise in this sampke the SNR can be inf.
        snr = radrow[0, 0, 0, 2]
        assert snr > 1e6 or np.isnan(snr)

    def it_finds_off_center():
        n_cycles = 1
        cen = 21 / 2
        y = -1.0 + cen
        x = 1.0 + cen
        chcy_ims = np.zeros((1, n_cycles, 21, 21))
        imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 21), loc=XY(0, 0), center=False)
        radrow = radiometry_field_stack(
            chcy_ims,
            locs=np.array([[y, x]]),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((n_cycles)),
        )
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001

    def it_finds_way_off_center():
        n_cycles = 1
        cen = 21 / 2
        y = -2.2 + cen
        x = 3.7 + cen
        chcy_ims = np.zeros((1, n_cycles, 21, 21))
        imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 21), loc=XY(0, 0), center=False)
        radrow = radiometry_field_stack(
            chcy_ims,
            locs=np.array([[y, x]]),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((n_cycles)),
        )
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.010

    def it_finds_fractional_loc():
        n_cycles = 1
        cen = 21 / 2
        for y in np.linspace(cen - 2, cen + 2, 14):
            for x in np.linspace(cen - 2, cen + 2, 14):
                chcy_ims = np.zeros((1, n_cycles, 21, 21))
                imops.accum_inplace(
                    chcy_ims[0, 0], _peak(x, y, 21), loc=XY(0, 0), center=False
                )
                radrow = radiometry_field_stack(
                    chcy_ims,
                    locs=np.array([[y, x]]),
                    reg_psf=reg_psf,
                    focus_adjustment=np.ones((n_cycles)),
                )
                assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
                assert np.abs(radrow[0, 0, 0, 1]) < 0.001

    def it_finds_many_locs_with_jitter():
        n_cycles = 1
        chcy_ims = np.zeros((1, n_cycles, 128, 128))
        locs = [
            (y, x) for y in np.linspace(10, 120, 8) for x in np.linspace(10, 120, 8)
        ]
        locs = np.array(locs)
        locs = locs + np.random.uniform(-1, 1, size=locs.shape)
        for loc in locs:
            imops.accum_inplace(
                chcy_ims[0, 0], _peak(loc[1], loc[0], 128), loc=XY(0, 0), center=False
            )

        radmat = radiometry_field_stack(
            chcy_ims, locs=locs, reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles))
        )

        assert np.all(np.abs(radmat[:, :, :, 0] - 1000.0) < 0.01)
        assert np.all(np.abs(radmat[:, :, :, 1]) < 0.2)

    def it_finds_changes_over_cycles():
        n_cycles = 3
        cen = reg_psf.peak_mea / 2
        y = 0.0 + cen
        x = 0.0 + cen
        chcy_ims = np.zeros((1, n_cycles, reg_psf.peak_mea, reg_psf.peak_mea))

        imops.accum_inplace(
            chcy_ims[0, 0],
            _peak(x, y, reg_psf.peak_mea, 1000),
            loc=XY(0, 0),
            center=False,
        )
        imops.accum_inplace(
            chcy_ims[0, 1],
            _peak(x, y, reg_psf.peak_mea, 900),
            loc=XY(0, 0),
            center=False,
        )
        imops.accum_inplace(
            chcy_ims[0, 2],
            _peak(x, y, reg_psf.peak_mea, 100),
            loc=XY(0, 0),
            center=False,
        )

        radrow = radiometry_field_stack(
            chcy_ims,
            locs=np.array([[y, x]]),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((n_cycles)),
        )

        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001

        assert np.abs(radrow[0, 0, 1, 0] - 900.0) < 0.01
        assert np.abs(radrow[0, 0, 1, 1]) < 0.001

        assert np.abs(radrow[0, 0, 2, 0] - 100.0) < 0.01
        assert np.abs(radrow[0, 0, 2, 1]) < 0.001

    def it_adjusts_for_focus():
        reg_psf = RegPSF.fixture(im_mea=128, peak_mea=11)
        ch_i = 0
        sigma_x = reg_psf.params[ch_i, 0, 0, RegPSF.SIGMA_X]
        sigma_y = reg_psf.params[ch_i, 0, 0, RegPSF.SIGMA_Y]
        rho = reg_psf.params[ch_i, 0, 0, RegPSF.RHO]

        n_cycles = 3
        chcy_ims = np.zeros((1, n_cycles, 128, 128))
        focuses = [1.0, 0.8, 1.2]
        locs = []
        for cy_i, focus in enumerate(focuses):
            for loc_y in np.linspace(20, 110, 10):
                for loc_x in np.linspace(20, 110, 10):
                    peak_im = imops.gauss2_rho_form(
                        1000.0,
                        focus * sigma_x,
                        focus * sigma_y,
                        loc_y,
                        loc_x,
                        rho,
                        0.0,
                        128,
                    )
                    imops.accum_inplace(
                        chcy_ims[0, cy_i], peak_im, loc=XY(0, 0), center=False
                    )
                    locs += [(loc_y, loc_x)]

        locs = np.array(locs)
        radmat = radiometry_field_stack(
            chcy_ims, locs=locs, reg_psf=reg_psf, focus_adjustment=np.array(focuses)
        )

        assert np.all(np.abs(radmat[:, :, :, 0] - 1000.0) < 20.0)

    def it_returns_asr():
        reg_psf = RegPSF.fixture(im_mea=128, peak_mea=11)

        chcy_ims = np.zeros((1, 1, 128, 128))
        peak_im = imops.gauss2_rho_form(1.0, 1.7, 2.3, 5.5, 4.5, 0.3, 0.0, 11)
        imops.accum_inplace(chcy_ims[0, 0], peak_im, YX(30, 20), center=True)
        locs = np.array([[30.0, 20.0]])

        radmat = radiometry_field_stack(
            chcy_ims, locs=locs, reg_psf=reg_psf, focus_adjustment=np.ones((1,))
        )

        assert 2.34 < radmat[0, 0, 0, 3] < 2.36

    zest()
