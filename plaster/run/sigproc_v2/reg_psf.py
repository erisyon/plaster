import numpy as np
from itertools import product
from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import Gauss2Params
from plaster.tools.image import imops
from plaster.tools.schema import check


class RegPSF:
    """
    Regional Point Spread Function as a 2D Gaussian.

    A PSF only models the sigma_x, sigma_y and rho... that is, only
    the Gaussian parameters that affect the size and shape of the PSF
    but not the amplitude (which we normalize to 1) nor the
    shift which we assume is dead center nor the constant which we
    assume is zero (ie fully background subtracted)
    """

    SIGMA_X = 0
    SIGMA_Y = 1
    RHO = 2
    N_PARAMS = 3

    def __init__(self, peak_mea, n_divs=5):
        self.peak_mea = peak_mea
        self.n_divs = n_divs
        self.params = np.zeros((n_divs, n_divs, RegPSF.N_PARAMS))

    def render_one_reg(self, div_y, div_x, amp=1.0, frac_y=0.0, frac_x=0.0, const=0.0):
        assert 0 <= div_y < self.n_divs
        assert 0 <= div_x < self.n_divs
        assert 0 <= frac_x <= 1.0
        assert 0 <= frac_y <= 1.0
        im = imops.gauss2_rho_form(
            amp=1.0,
            std_x=self.params[div_y, div_x, self.SIGMA_X],
            std_y=self.params[div_y, div_x, self.SIGMA_Y],
            pos_x=self.peak_mea / 2 + frac_x,
            pos_y=self.peak_mea / 2 + frac_y,
            rho=self.params[div_y, div_x, self.RHO],
            const=const,
            mea=self.peak_mea,
        )

        # Normalize to get an AUC exactly equal to amp
        return amp * im / np.sum(im)

    def render(self):
        psf_ims = np.zeros((self.n_divs, self.n_divs, self.peak_mea, self.peak_mea))
        for y, x in product(range(self.n_divs), range(self.n_divs)):
            psf_ims[y, x] = self.render_one_reg(y, x)
        return psf_ims

    def _fit(self, im, y, x):
        check.array_t(im, ndim=2, is_square=True)
        if np.sum(im) > 0:
            fit_params, _ = imops.fit_gauss2(im)
            self.params[y, x, :] = (
                fit_params[Gauss2Params.SIGMA_X],
                fit_params[Gauss2Params.SIGMA_Y],
                fit_params[Gauss2Params.RHO],
            )
        else:
            self.params[y, x, :] = 0

    @classmethod
    def from_psf_ims(cls, psf_ims):
        """
        Fit to a Gaussian, remove bias, and resample
        """
        check.array_t(psf_ims, ndim=4)
        divs_y, divs_x, peak_mea_h, peak_mea_w = psf_ims.shape
        assert divs_y == divs_x
        assert peak_mea_h == peak_mea_w
        reg_psf = cls(peak_mea=peak_mea_h, n_divs=divs_y)
        for y in range(divs_y):
            for x in range(divs_x):
                reg_psf._fit(psf_ims[y, x], y, x)

        return reg_psf

    @classmethod
    def from_array(cls, arr):
        check.array_t(arr, ndim=3)
        divs_y, divs_x, n_gauss_params = arr.shape
        assert n_gauss_params == cls.N_PARAMS

        hard_coded_peak_mea = 11
        # This is a HACK until I rebuild the calibration classes

        reg_psf = cls(peak_mea=hard_coded_peak_mea, n_divs=divs_y)
        reg_psf.params = arr

        return reg_psf

    @classmethod
    def fixture(cls, peak_mea=11, n_divs=5, sig_x=1.8, sig_y=1.8, rho=0.0):
        reg_psf = cls(peak_mea=peak_mea, n_divs=n_divs)
        reg_psf.params[:, :, 0] = sig_x
        reg_psf.params[:, :, 1] = sig_y
        reg_psf.params[:, :, 2] = rho
        return reg_psf


def approximate_psf():
    """
    Return a zero-centered AUC=1.0 2D Gaussian for peak finding
    """
    std = 1.5  # This needs to be tuned and may be instrument dependent
    mea = 11
    kern = imops.gauss2_rho_form(
        amp=1.0,
        std_x=std,
        std_y=std,
        pos_x=mea // 2,
        pos_y=mea // 2,
        rho=0.0,
        const=0.0,
        mea=mea,
    )
    return kern - np.mean(kern)
