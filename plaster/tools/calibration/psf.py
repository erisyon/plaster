import numpy as np
from itertools import product
from plaster.tools.image import imops


class Gauss2Params:
    AMP = 0
    SIGMA_X = 1
    SIGMA_Y = 2
    CENTER_X = 3
    CENTER_Y = 4
    RHO = 5
    OFFSET = 6
    N_PARAMS = 7  # Number above this point


class RegPSF:
    """
    Regional Point Spread Function as a 2D Gaussian.
    Enforces that amplitude is 1.0, const is 0.0, and center is exactly the center of the buffer.
    """

    def __init__(self, peak_mea, n_divs=5):
        self.peak_mea = peak_mea
        self.n_divs = n_divs
        self.params = np.zeros((n_divs, n_divs, Gauss2Params.N_PARAMS))

    def set_reg(self, y, x, params):
        assert len(params) == 3
        self.params[y, x, Gauss2Params.SIGMA_X] = params[0]
        self.params[y, x, Gauss2Params.SIGMA_Y] = params[1]
        self.params[y, x, Gauss2Params.RHO] = params[2]

    def render_one_reg(self, div_y, div_x, amp=1.0, frac_y=0.0, frac_x=0.0, const=0.0):
        assert 0 <= div_y < self.n_divs
        assert 0 <= div_x < self.n_divs
        assert 0 <= frac_x <= 1.0
        assert 0 <= frac_y <= 1.0
        im = imops.gauss2_rho_form(
            amp=1.0,
            std_x=self.params[div_y, div_x, Gauss2Params.SIGMA_X],
            std_y=self.params[div_y, div_x, Gauss2Params.SIGMA_Y],
            pos_x=self.peak_mea / 2 + frac_x,
            pos_y=self.peak_mea / 2 + frac_y,
            rho=self.params[div_y, div_x, Gauss2Params.RHO],
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

    def new_sigmas(self, new_sigma_x, new_sigma_y):
        """
        Make a copy of this RegPSF with the sigma changes
        """

        new_reg_psf = RegPSF(self.peak_mea, self.n_divs)
        new_reg_psf.params = self.params.copy()
        new_reg_psf.params[:, :, Gauss2Params.SIGMA_X] = new_sigma_x
        new_reg_psf.params[:, :, Gauss2Params.SIGMA_Y] = new_sigma_y
        return new_reg_psf
