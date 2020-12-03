import numpy as np
from itertools import product
from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import Gauss2Params
from plaster.tools.image import imops
from plaster.tools.schema import check
from plaster.tools.log.log import debug
from scipy import interpolate


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

    def _init_interpolation(self):
        if self.interp_sig_x_fn is None:
            center = self.im_mea / self.n_divs / 2.0
            coords = np.linspace(center, self.im_mea - center, self.n_divs)
            xx, yy = np.meshgrid(coords, coords)
            self.interp_sig_x_fn = interpolate.interp2d(
                xx, yy, self.params[:, :, RegPSF.SIGMA_X], kind="cubic"
            )
            self.interp_sig_y_fn = interpolate.interp2d(
                xx, yy, self.params[:, :, RegPSF.SIGMA_Y], kind="cubic"
            )
            self.interp_rho_fn = interpolate.interp2d(
                xx, yy, self.params[:, :, RegPSF.RHO], kind="cubic"
            )

    def __init__(self, im_mea, peak_mea, n_divs):
        """
        Arguments:
            im_mea: tuple (height / width) of the raw images before alignment
            peak_mea: number of pixel (height/ width) representing the peak
            n_divs: number of spatial divisions (height / width)
        """
        self.im_mea = im_mea
        self.peak_mea = peak_mea
        self.n_divs = n_divs
        self.params = np.zeros((n_divs, n_divs, RegPSF.N_PARAMS))
        self.interp_sig_x_fn = None
        self.interp_sig_y_fn = None
        self.interp_rho_fn = None

    def render_one_reg(self, div_y, div_x, amp=1.0, frac_y=0.0, frac_x=0.0, const=0.0):
        assert 0 <= div_y < self.n_divs
        assert 0 <= div_x < self.n_divs
        assert 0 <= frac_x <= 1.0
        assert 0 <= frac_y <= 1.0
        im = imops.gauss2_rho_form(
            amp=1.0,
            std_x=self.params[div_y, div_x, self.SIGMA_X],
            std_y=self.params[div_y, div_x, self.SIGMA_Y],
            # Note that the following must be integer divides because the
            # fractional component is relative to the lower-left corner (origin)
            pos_x=self.peak_mea // 2 + frac_x,
            pos_y=self.peak_mea // 2 + frac_y,
            rho=self.params[div_y, div_x, self.RHO],
            const=const,
            mea=self.peak_mea,
        )

        # Normalize to get an AUC exactly equal to amp
        return amp * im / np.sum(im)

    def render_at_loc(self, loc, amp=1.0, const=0.0, focus=1.0):
        self._init_interpolation()
        loc_x = loc[1]
        loc_y = loc[0]
        sig_x = self.interp_sig_x_fn(loc_x, loc_y)[0]
        sig_y = self.interp_sig_y_fn(loc_x, loc_y)[0]
        rho = self.interp_rho_fn(loc_x, loc_y)[0]

        half_mea = self.peak_mea / 2.0

        corner_x = np.floor(loc_x - half_mea + 0.5)
        corner_y = np.floor(loc_y - half_mea + 0.5)
        center_x = loc_x - corner_x
        center_y = loc_y - corner_y
        im = imops.gauss2_rho_form(
            amp=amp,
            std_x=sig_x * focus,
            std_y=sig_y * focus,
            pos_x=center_x,
            pos_y=center_y,
            rho=rho,
            const=const,
            mea=self.peak_mea,
        )

        return im, (corner_y, corner_x)

    def render(self):
        psf_ims = np.zeros((self.n_divs, self.n_divs, self.peak_mea, self.peak_mea))
        for y, x in product(range(self.n_divs), range(self.n_divs)):
            psf_ims[y, x] = self.render_one_reg(y, x)
        return psf_ims

    def sample_params(self):
        self._init_interpolation()
        space = np.linspace(0, self.im_mea, 6)
        n_samples = len(space) ** 2
        samples = np.zeros((n_samples, 5))
        i = 0
        for y in space:
            for x in space:
                sig_x = self.interp_sig_x_fn(x, y)
                sig_y = self.interp_sig_y_fn(x, y)
                rho = self.interp_rho_fn(x, y)
                samples[i, :] = (x, y, sig_x, sig_y, rho)
                i += 1
        return samples

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
    def from_psf_ims(cls, im_mea, psf_ims):
        """
        Fit to a Gaussian, remove bias, and resample
        """
        check.array_t(psf_ims, ndim=4)
        divs_y, divs_x, peak_mea_h, peak_mea_w = psf_ims.shape
        assert divs_y == divs_x
        assert peak_mea_h == peak_mea_w
        reg_psf = cls(im_mea=im_mea, peak_mea=peak_mea_h, n_divs=divs_y)
        for y in range(divs_y):
            for x in range(divs_x):
                reg_psf._fit(psf_ims[y, x], y, x)

        return reg_psf

    @classmethod
    def from_array(cls, im_mea, peak_mea, arr):
        check.array_t(arr, ndim=3)
        divs_y, divs_x, n_gauss_params = arr.shape
        assert divs_y == divs_x
        assert n_gauss_params == cls.N_PARAMS

        reg_psf = cls(im_mea=im_mea, peak_mea=peak_mea, n_divs=divs_y)
        reg_psf.params = arr

        return reg_psf

    @classmethod
    def fixture(cls, im_mea=512, peak_mea=15, n_divs=5, sig_x=1.8, sig_y=1.8, rho=0.0):
        reg_psf = cls(im_mea=im_mea, peak_mea=peak_mea, n_divs=n_divs)
        reg_psf.params[:, :, 0] = sig_x
        reg_psf.params[:, :, 1] = sig_y
        reg_psf.params[:, :, 2] = rho
        return reg_psf

    @classmethod
    def fixture_variable(cls, im_mea=512, peak_mea=15, n_divs=5):
        reg_psf = cls(im_mea=im_mea, peak_mea=peak_mea, n_divs=n_divs)
        for y in range(n_divs):
            cy = y - n_divs // 2
            for x in range(n_divs):
                cx = x - n_divs // 2
                reg_psf.params[y, x, RegPSF.SIGMA_X] = 1.5 + 0.05 * np.abs(cx * cy)
                reg_psf.params[y, x, RegPSF.SIGMA_Y] = 1.5 + 0.10 * np.abs(cx * cy)
                reg_psf.params[y, x, RegPSF.RHO] = 0.02 * cx * cy
        return reg_psf

    @classmethod
    def fixture_radical(cls, im_mea=512, peak_mea=15, n_divs=5):
        reg_psf = cls(im_mea=im_mea, peak_mea=peak_mea, n_divs=n_divs)
        for y in range(n_divs):
            cy = y - n_divs // 2
            for x in range(n_divs):
                cx = x - n_divs // 2
                reg_psf.params[y, x, RegPSF.SIGMA_X] = 6 + 0.1 * np.abs(cx * cy)
                reg_psf.params[y, x, RegPSF.SIGMA_Y] = 2 + 0.2 * np.abs(cx * cy)
                reg_psf.params[y, x, RegPSF.RHO] = 0.10 * cx * cy
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
