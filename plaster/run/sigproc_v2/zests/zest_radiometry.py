from zest import zest
from plaster.tools.image import imops
from plaster.tools.image.coord import XY, YX
from plaster.run.sigproc_v2.reg_psf import RegPSF
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
        n_cycles = 1
        cen = 11 / 2
        y = 0.0 + cen
        x = 0.0 + cen
        chcy_ims = np.zeros((1, n_cycles, 11, 11))
        imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 11), loc=XY(0, 0), center=False)
        radrow = radiometry_field_stack(
            chcy_ims,
            locs=np.array([[y, x]]),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((n_cycles)),
        )
        # print(f"{y:8.2f} {x:8.2f} {radrow[0, 0, 0, 0]:8.2f} {radrow[0, 0, 0, 1]:8.2f}")
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001
        assert np.abs(radrow[0, 0, 0, 2]) > 1e6

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
        x = 5.7 + cen
        chcy_ims = np.zeros((1, n_cycles, 21, 21))
        imops.accum_inplace(chcy_ims[0, 0], _peak(x, y, 21), loc=XY(0, 0), center=False)
        radrow = radiometry_field_stack(
            chcy_ims,
            locs=np.array([[y, x]]),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((n_cycles)),
        )
        # np.save("./_test.npy", chcy_ims[0, 0])
        # print(f"{y:8.2f} {x:8.2f} {radrow[0, 0, 0, 0]:8.2f} {radrow[0, 0, 0, 1]:8.2f}")
        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001

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
                # print(f"{y:8.2f} {x:8.2f} {radrow[0,0,0,0]:8.2f} {radrow[0,0,0,1]:8.2f}")
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
        # print(f"{y:8.2f} {x:8.2f} {radrow[0,0,0,0]:8.2f} {radrow[0,0,0,1]:8.2f}")
        assert np.all(np.abs(radmat[:, :, :, 0] - 1000.0) < 0.01)
        assert np.all(np.abs(radmat[:, :, :, 1]) < 0.01)

    def it_finds_changes_over_cycles():
        n_cycles = 3
        cen = 11 / 2
        y = 0.0 + cen
        x = 0.0 + cen
        chcy_ims = np.zeros((1, n_cycles, 11, 11))

        imops.accum_inplace(
            chcy_ims[0, 0], _peak(x, y, 11, 1000), loc=XY(0, 0), center=False
        )
        imops.accum_inplace(
            chcy_ims[0, 1], _peak(x, y, 11, 900), loc=XY(0, 0), center=False
        )
        imops.accum_inplace(
            chcy_ims[0, 2], _peak(x, y, 11, 100), loc=XY(0, 0), center=False
        )

        radrow = radiometry_field_stack(
            chcy_ims,
            locs=np.array([[y, x]]),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((n_cycles)),
        )
        # print(f"{y:8.2f} {x:8.2f} {radrow[0, 0, 0, 0]:8.2f} {radrow[0, 0, 0, 1]:8.2f}")
        # print(f"{y:8.2f} {x:8.2f} {radrow[0, 0, 1, 0]:8.2f} {radrow[0, 0, 1, 1]:8.2f}")
        # print(f"{y:8.2f} {x:8.2f} {radrow[0, 0, 2, 0]:8.2f} {radrow[0, 0, 2, 1]:8.2f}")

        assert np.abs(radrow[0, 0, 0, 0] - 1000.0) < 0.01
        assert np.abs(radrow[0, 0, 0, 1]) < 0.001

        assert np.abs(radrow[0, 0, 1, 0] - 900.0) < 0.01
        assert np.abs(radrow[0, 0, 1, 1]) < 0.001

        assert np.abs(radrow[0, 0, 2, 0] - 100.0) < 0.01
        assert np.abs(radrow[0, 0, 2, 1]) < 0.001

    def it_uses_appropriate_regional_psf():
        peak_ws = [1.8, 1.8, 2.4]
        peak_rhos = [0.0, 0.5, 0.1]

        reg_psf = RegPSF((128, 128), 11, 2)
        reg_psf.params[0, 0] = (peak_ws[0], 1.8, peak_rhos[0])
        reg_psf.params[1, 0] = (peak_ws[1], 1.8, peak_rhos[1])
        reg_psf.params[1, 1] = (peak_ws[2], 1.8, peak_rhos[2])

        n_cycles = 1
        chcy_ims = np.zeros((1, n_cycles, 128, 128))
        locs = np.array([(20.0, 20.0), (100.0, 20.0), (100.0, 100.0),])
        for loc, peak_w, peak_rho in zip(locs, peak_ws, peak_rhos):
            peak_im = imops.gauss2_rho_form(
                1000.0, peak_w, 1.8, loc[1], loc[0], peak_rho, 0.0, 128
            )
            imops.accum_inplace(chcy_ims[0, 0], peak_im, loc=XY(0, 0), center=False)

        radmat = radiometry_field_stack(
            chcy_ims, locs=locs, reg_psf=reg_psf, focus_adjustment=np.ones((n_cycles))
        )
        assert np.all(np.abs(radmat[:, :, :, 0] - 1000.0) < 0.01)
        assert np.all(np.abs(radmat[:, :, :, 1]) < 0.001)

    def it_adjusts_for_focus():
        peak_ws = [1.8, 1.8, 2.4]
        peak_rhos = [0.0, 0.5, 0.1]

        reg_psf = RegPSF((128, 128), 11, 2)
        reg_psf.params[0, 0] = (peak_ws[0], 1.8, peak_rhos[0])
        reg_psf.params[1, 0] = (peak_ws[1], 1.8, peak_rhos[1])
        reg_psf.params[1, 1] = (peak_ws[2], 1.8, peak_rhos[2])

        n_cycles = 3
        chcy_ims = np.zeros((1, n_cycles, 128, 128))
        locs = np.array([(20.0, 20.0), (100.0, 20.0), (100.0, 100.0),])
        focuses = [1.0, 0.8, 1.2]
        for cy_i, focus in enumerate(focuses):
            for loc, peak_w, peak_rho in zip(locs, peak_ws, peak_rhos):
                peak_im = imops.gauss2_rho_form(
                    1000.0,
                    focus * peak_w,
                    focus * 1.8,
                    loc[1],
                    loc[0],
                    peak_rho,
                    0.0,
                    128,
                )
                imops.accum_inplace(
                    chcy_ims[0, cy_i], peak_im, loc=XY(0, 0), center=False
                )

        radmat = radiometry_field_stack(
            chcy_ims, locs=locs, reg_psf=reg_psf, focus_adjustment=np.array(focuses)
        )
        # np.save("./_test.npy", chcy_ims[0])
        # print(f"{radmat[0, 0, 0, 0]:8.2f} {radmat[0, 0, 0, 1]:8.2f}")
        # print(f"{radmat[0, 0, 1, 0]:8.2f} {radmat[0, 0, 1, 1]:8.2f}")
        # print(f"{radmat[0, 0, 2, 0]:8.2f} {radmat[0, 0, 2, 1]:8.2f}")
        #
        # print(f"{radmat[1, 0, 0, 0]:8.2f} {radmat[1, 0, 0, 1]:8.2f}")
        # print(f"{radmat[1, 0, 1, 0]:8.2f} {radmat[1, 0, 1, 1]:8.2f}")
        # print(f"{radmat[1, 0, 2, 0]:8.2f} {radmat[1, 0, 2, 1]:8.2f}")
        #
        # print(f"{radmat[2, 0, 0, 0]:8.2f} {radmat[2, 0, 0, 1]:8.2f}")
        # print(f"{radmat[2, 0, 1, 0]:8.2f} {radmat[2, 0, 1, 1]:8.2f}")
        # print(f"{radmat[2, 0, 2, 0]:8.2f} {radmat[2, 0, 2, 1]:8.2f}")
        assert np.all(np.abs(radmat[:, :, :, 0] - 1000.0) < 0.01)
        assert np.all(np.abs(radmat[:, :, :, 1]) < 0.001)

    # def it_returns_asr():
    #     raise NotImplementedError

    zest()
