import base64
from itertools import product

import numpy as np
import re
from munch import Munch
from scipy import interpolate

from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import Gauss2Params
from plaster.tools.image import imops
from plaster.tools.utils import utils
from plumbum import local
from enum import Enum
from dataclasses import dataclass
from plaster.tools.log.log import debug
from plaster.tools.schema import check


class RegIllum:
    """
    Regional Illumination Balance.
    """

    def __init__(self, n_channels, im_mea, n_divs):
        check.t(n_channels, int)
        check.t(im_mea, int)
        self.n_channels = n_channels
        self.im_mea = im_mea
        self.n_divs = n_divs
        self.reg_illum = np.zeros((n_channels, n_divs, n_divs))

    def set(self, ch_i, values):
        check.array_t(values, shape=(self.n_divs, self.n_divs))
        self.reg_illum[ch_i] = values

    def interp(self, ch_i):
        """
        Expand the regional balance into a full image
        """
        return imops.interp(self.reg_illum[ch_i], (self.im_mea, self.im_mea))

    @classmethod
    def identity(cls, n_channels, im_mea):
        n_divs = 5
        reg_illum = RegIllum(n_channels, im_mea, n_divs)
        reg_illum.reg_illum = np.ones((n_channels, n_divs, n_divs))
        return reg_illum


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

    def _init_interpolation(self, ch_i):
        if self.interp_sig_x_fn[ch_i] is None:
            self.params.flags.writeable = False  # Prevent changes thus enabling caching
            center = self.im_mea / self.n_divs / 2.0
            coords = np.linspace(center, self.im_mea - center, self.n_divs)
            xx, yy = np.meshgrid(coords, coords)
            self.interp_sig_x_fn[ch_i] = interpolate.interp2d(
                xx, yy, self.params[ch_i, :, :, RegPSF.SIGMA_X], kind="cubic"
            )
            self.interp_sig_y_fn[ch_i] = interpolate.interp2d(
                xx, yy, self.params[ch_i, :, :, RegPSF.SIGMA_Y], kind="cubic"
            )
            self.interp_rho_fn[ch_i] = interpolate.interp2d(
                xx, yy, self.params[ch_i, :, :, RegPSF.RHO], kind="cubic"
            )

    def __init__(self, n_channels, im_mea, peak_mea, n_divs):
        """
        Arguments:
            im_mea: tuple (height or width) of the raw images before alignment
            peak_mea: number of pixel (height or width) representing the peak
            n_divs: number of spatial divisions (height or width)
        """
        check.t(n_channels, int)
        check.t(im_mea, int)
        check.t(peak_mea, int)
        check.t(n_divs, int)
        self.n_channels = n_channels
        self.im_mea = im_mea
        self.peak_mea = peak_mea
        self.n_divs = n_divs
        self.params = np.zeros((n_channels, n_divs, n_divs, RegPSF.N_PARAMS))
        self.interp_sig_x_fn = [None] * n_channels
        self.interp_sig_y_fn = [None] * n_channels
        self.interp_rho_fn = [None] * n_channels
        self._grid_cache = None
        self._grid_hash = None

        # ZBS: I don't love this modal channel selection but for now
        # it gets me over a hump of various code that expectes RegPSF
        # to be for a single channel
        self._selected_ch_i = None

    def select_ch(self, ch_i):
        self._selected_ch_i = ch_i

    def get_params(self, y, x, param):
        return self.params[self._selected_ch_i, y, x, param]

    def render_one_reg(
        self, ch_i, div_y, div_x, amp=1.0, frac_y=0.0, frac_x=0.0, const=0.0
    ):
        if ch_i is None:
            ch_i = self._selected_ch_i
        assert 0 <= ch_i < self.n_channels
        assert 0 <= div_y < self.n_divs
        assert 0 <= div_x < self.n_divs
        assert 0 <= frac_x <= 1.0
        assert 0 <= frac_y <= 1.0
        im = imops.gauss2_rho_form(
            amp=1.0,
            std_x=self.params[ch_i, div_y, div_x, self.SIGMA_X],
            std_y=self.params[ch_i, div_y, div_x, self.SIGMA_Y],
            # Note that the following must be integer divides because the
            # fractional component is relative to the lower-left corner (origin)
            pos_x=self.peak_mea // 2 + frac_x,
            pos_y=self.peak_mea // 2 + frac_y,
            rho=self.params[ch_i, div_y, div_x, self.RHO],
            const=const,
            mea=self.peak_mea,
        )

        # Normalize to get an AUC exactly equal to amp
        return amp * im / np.sum(im)

    def render_at_loc(self, ch_i, loc, amp=1.0, const=0.0, focus=1.0):
        if ch_i is None:
            ch_i = self._selected_ch_i
        assert 0 <= ch_i < self.n_channels
        self._init_interpolation(ch_i)
        loc_x = loc[1]
        loc_y = loc[0]
        sig_x = self.interp_sig_x_fn[ch_i](loc_x, loc_y)[0]
        sig_y = self.interp_sig_y_fn[ch_i](loc_x, loc_y)[0]
        rho = self.interp_rho_fn[ch_i](loc_x, loc_y)[0]

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
        psf_ims = np.zeros(
            (self.n_channels, self.n_divs, self.n_divs, self.peak_mea, self.peak_mea)
        )
        for y, x in product(range(self.n_divs), range(self.n_divs)):
            for ch_i in range(self.n_channels):
                psf_ims[ch_i, y, x] = self.render_one_reg(ch_i, y, x)
        if ch_i is not None:
            psf_ims = psf_ims[ch_i]
        return psf_ims

    def sample_params(self, ch_i, n_divs=6):
        if ch_i is None:
            ch_i = self._selected_ch_i
        self._init_interpolation(self.n_channels)
        space = np.linspace(0, self.im_mea, n_divs)
        n_samples = len(space) ** 2
        samples = np.zeros((n_samples, 5))
        i = 0
        for y in space:
            for x in space:
                sig_x = self.interp_sig_x_fn[ch_i](x, y)
                sig_y = self.interp_sig_y_fn[ch_i](x, y)
                rho = self.interp_rho_fn[ch_i](x, y)
                samples[i, :] = (x, y, sig_x, sig_y, rho)
                i += 1
        return samples

    def sample_params_grid(self, ch_i, n_divs=6):
        # TODO: Optimize to avoid the python double loop. Numpy
        #   Something is wrong because when I try this in a notebook it is instant
        #   but here is taking almost 0.5 sec?
        if ch_i is None:
            ch_i = self._selected_ch_i
        self_hash = hash((self, n_divs))
        if self_hash == self._grid_hash:
            return self._grid_cache

        self._init_interpolation(ch_i)
        space = np.linspace(0, self.im_mea, n_divs)
        samples = np.zeros((n_divs, n_divs, 3))
        for yi, y in enumerate(space):
            for xi, x in enumerate(space):
                sig_x = self.interp_sig_x_fn[ch_i](x, y)
                sig_y = self.interp_sig_y_fn[ch_i](x, y)
                rho = self.interp_rho_fn[ch_i](x, y)
                samples[yi, xi, :] = (sig_x, sig_y, rho)

        self._grid_cache = samples
        self._grid_hash = self_hash
        return samples

    def _fit(self, im, ch_i, y, x):
        if ch_i is None:
            ch_i = self._selected_ch_i
        check.array_t(im, ndim=2, is_square=True)
        if np.sum(im) > 0:
            fit_params, _ = imops.fit_gauss2(im)
            self.params[ch_i, y, x, :] = (
                fit_params[Gauss2Params.SIGMA_X],
                fit_params[Gauss2Params.SIGMA_Y],
                fit_params[Gauss2Params.RHO],
            )
        else:
            self.params[ch_i, y, x, :] = 0

    @classmethod
    def from_psf_ims(cls, im_mea, psf_ims):
        """
        Fit to a Gaussian for one-channel
        """
        check.array_t(psf_ims, ndim=4)
        divs_y, divs_x, peak_mea_h, peak_mea_w = psf_ims.shape
        assert divs_y == divs_x
        assert peak_mea_h == peak_mea_w
        reg_psf = cls(n_channels=1, im_mea=im_mea, peak_mea=peak_mea_h, n_divs=divs_y)
        for y in range(divs_y):
            for x in range(divs_x):
                reg_psf._fit(psf_ims[y, x], ch_i=0, y=y, x=x)

        return reg_psf

    @classmethod
    def from_array(cls, im_mea, peak_mea, arr):
        check.array_t(arr, ndim=4)
        n_channels, divs_y, divs_x, n_gauss_params = arr.shape
        assert divs_y == divs_x
        assert n_gauss_params == cls.N_PARAMS

        reg_psf = cls(
            n_channels=n_channels, im_mea=im_mea, peak_mea=peak_mea, n_divs=divs_y
        )
        reg_psf.params = arr

        return reg_psf

    @classmethod
    def from_channel_reg_psfs(cls, reg_psfs):
        """
        ZBS: I'm not happy with this pattern of assembling reg_psfs from channels
        but it will do for now and it is minimal changes to existing code.
        """
        n_channels = len(reg_psfs)
        check.list_t(reg_psfs, RegPSF)
        reg_psf = cls(
            n_channels, reg_psfs[0].im_mea, reg_psfs[0].peak_mea, reg_psfs[0].n_divs
        )
        for ch_i, ch_reg_psf in enumerate(reg_psfs):
            assert ch_reg_psf.im_mea == reg_psf.im_mea
            assert ch_reg_psf.peak_mea == reg_psf.peak_mea
            assert ch_reg_psf.n_divs == reg_psf.n_divs
            assert ch_reg_psf.n_channels == 1
            reg_psf.params[ch_i] = ch_reg_psf.params[0]
        return reg_psf

    @classmethod
    def fixture(
        cls,
        n_channels=1,
        im_mea=512,
        peak_mea=15,
        n_divs=5,
        sig_x=1.8,
        sig_y=1.8,
        rho=0.0,
    ):
        reg_psf = cls(
            n_channels=n_channels, im_mea=im_mea, peak_mea=peak_mea, n_divs=n_divs
        )
        reg_psf.params[:, :, :, 0] = sig_x
        reg_psf.params[:, :, :, 1] = sig_y
        reg_psf.params[:, :, :, 2] = rho
        return reg_psf

    @classmethod
    def fixture_variable(cls, n_channels=1, im_mea=512, peak_mea=15, n_divs=5):
        reg_psf = cls(
            n_channels=n_channels, im_mea=im_mea, peak_mea=peak_mea, n_divs=n_divs
        )
        for y in range(n_divs):
            cy = y - n_divs // 2
            for x in range(n_divs):
                cx = x - n_divs // 2
                reg_psf.params[:, y, x, RegPSF.SIGMA_X] = 1.5 + 0.05 * np.abs(cx * cy)
                reg_psf.params[:, y, x, RegPSF.SIGMA_Y] = 1.5 + 0.10 * np.abs(cx * cy)
                reg_psf.params[:, y, x, RegPSF.RHO] = 0.02 * cx * cy
        return reg_psf


class CalibIdentity:
    def __init__(self, id):
        check.t(id, (str, CalibIdentity))
        if isinstance(id, CalibIdentity):
            self.id = id.id
        else:
            self.id = id

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def validate(self):
        if self.id is None or self.id == "" or not isinstance(self.id, str):
            raise TypeError(f"CalibIndentity '{self.id}' not valid")


class CalibType(Enum):
    REG_PSF = 1
    REG_ILLUM = 2


@dataclass
class CalibRecord:
    """A record for any calibration object, see CalibBag"""

    calib_identity: CalibIdentity
    calib_type: CalibType
    value: object


class Calib:
    """A container of CalibRecords"""

    def __init__(self, recs=None):
        if recs is None:
            self.recs = []
        else:
            check.list_t(recs, CalibRecord)
            self.recs = recs
        self.validate()

    def validate(self):
        # TODO: add checks for duplicates, etc, raise on errors
        pass

    def keep_identity(self, calib_identity: CalibIdentity):
        check.t(calib_identity, CalibIdentity)
        self.recs = [rec for rec in self.recs if rec.calib_identity == calib_identity]
        return self

    def set_identity(self, id: str):
        check.t(id, str)
        calib_identity = CalibIdentity(id)
        for rec in self.recs:
            rec.calib_identity = calib_identity
        return self

    def has_records(self):
        return len(self.recs) > 0

    def find_rec_of_type(self, calib_type: CalibType):
        rec = utils.filt_first(self.recs, lambda rec: rec.calib_type == calib_type)
        if rec is not None:
            return rec.value
        return None

    def reg_psf(self, ch_i=None) -> RegPSF:
        rec = self.find_rec_of_type(CalibType.REG_PSF)
        if rec is None:
            raise KeyError("No object of CalibType.REG_PSF was found in calib")
        check.t(rec, RegPSF)
        if ch_i is not None:
            rec.select_ch(ch_i)  # See other notes, I don't like this solution
        return rec

    def add_reg_psf(self, reg_psf: RegPSF, calib_identity: CalibIdentity = None):
        check.t(reg_psf, RegPSF)

        if self.find_rec_of_type(CalibType.REG_PSF) is not None:
            raise KeyError("Duplicate object of CalibType.REG_PSF was found in calib")

        self.recs += [
            CalibRecord(
                calib_identity=calib_identity,
                calib_type=CalibType.REG_PSF,
                value=reg_psf,
            )
        ]

        self.validate()

    def reg_illum(self) -> RegIllum:
        rec = self.find_rec_of_type(CalibType.REG_ILLUM)
        if rec is None:
            raise KeyError("No object of CalibType.REG_ILLUM was found in calib")
        check.t(rec, RegIllum)
        return rec

    def add_reg_illum(self, reg_illum: RegIllum, calib_identity: CalibIdentity = None):
        check.t(reg_illum, RegIllum)

        if self.find_rec_of_type(CalibType.REG_ILLUM) is not None:
            raise KeyError("Duplicate object of CalibType.REG_ILLUM was found in calib")

        self.recs += [
            CalibRecord(
                calib_identity=calib_identity,
                calib_type=CalibType.REG_ILLUM,
                value=reg_illum,
            )
        ]

        self.validate()

    def save_file(self, path):
        _identity = None
        for rec in self.recs:
            if _identity is not None and rec.calib_identity != _identity:
                raise ValueError(
                    "All calib records must have the same identity during save"
                )
            if rec.calib_identity is None:
                raise ValueError("calib identity not specified in calib save")
            rec.calib_identity.validate()
            _identity = rec.calib_identity
        utils.pickle_save(path, self.recs)

    @classmethod
    def load_file(cls, path: str, id):
        check.t(id, (str, CalibIdentity))
        identity = CalibIdentity(id)
        _recs = utils.pickle_load(path)
        calib = cls(_recs)
        calib.keep_identity(identity)
        return calib


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
