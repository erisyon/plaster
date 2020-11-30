import numpy as np
import cv2
import random
import copy
import math
from plaster.run.ims_import.ims_import_worker import OUTPUT_NP_TYPE
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.sigproc_v2.reg_psf import RegPSF
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, ROI, WH, XY, YX
from plaster.tools.log.log import debug, important
from plaster.tools.utils import utils
from plaster.tools.utils.tmp import tmp_folder
from plaster.tools.schema import check
from plumbum import local

# see comment below, above "PeaksModelPSF" regarding why this is commented out
# from plaster.run.sigproc_v2.psf_sample import psf_sample


class Synth:
    """
    Generate synthetic images for testing.

    This system is organized so that synthetic image(s) is
    delayed until the render() command is called. This allows
    for "reaching in" to the state and messing with it.

    Example, suppose that in some strange test you need to
    have a position of a certain peak location in very specific
    places for the test. To prevent a proliferation of one-off
    methods in this class, the idea is that you can use the
    method that creates two peaks and then "reach in" to
    tweak the positions directly before render.

    Examples:
        with Synth() as s:
            p = PeaksModelGaussian()
            p.locs_randomize()
            CameraModel(100, 2)
            s.render_chcy()

    """

    synth = None

    def __init__(
        self,
        n_fields=1,
        n_channels=1,
        n_cycles=1,
        dim=(512, 512),
        save_as=None,
        overwrite=False,
    ):
        self.n_fields = n_fields
        self.n_channels = n_channels
        self.n_cycles = n_cycles
        self.dim = dim
        self.save_as = save_as
        self.models = []
        self.aln_offsets = np.random.uniform(-20, 20, size=(self.n_cycles, 2))
        self.aln_offsets[0] = (0, 0)
        if not overwrite:
            assert Synth.synth is None
        Synth.synth = self

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        Synth.synth = None
        if exception_type is not None:
            raise exception_type(exception_value)
        ims = self.render_flchcy()
        self._save_np(ims, "ims")

    def zero_aln_offsets(self):
        self.aln_offsets = np.zeros((self.n_cycles, 2))

    def add_model(self, model):
        self.models += [model]

    def _save_np(self, arr, name):
        if self.save_as is not None:
            save_as = local.path(self.save_as) + f"_{name}"
            np.save(save_as, arr)
            important(f"Wrote synth image to {save_as}.npy")

    def render_chcy(self, fl_i=0):
        """
        Returns only chcy_ims (first field)
        """
        ims = np.zeros((self.n_channels, self.n_cycles, *self.dim))
        for ch_i in np.arange(self.n_channels):
            for cy_i in np.arange(self.n_cycles):
                im = ims[ch_i, cy_i]
                for model in self.models:
                    model.render(im, fl_i, ch_i, cy_i)

                ims[ch_i, cy_i] = imops.sub_pixel_shift(im, self.aln_offsets[cy_i])

        return ims

    def render_flchcy(self):
        flchcy_ims = np.zeros(
            (self.n_fields, self.n_channels, self.n_cycles, *self.dim)
        )
        for fl_i in range(self.n_fields):
            flchcy_ims[fl_i] = self.render_chcy()
        return flchcy_ims

    def scale_peaks_by_max(self):
        """
        For some tests it is nice to know that the max brightness of a peak
        instead of the area under the curve.
        """
        self.peak_ims = [peak_im / np.max(peak_im) for peak_im in self.peak_ims]


class BaseSynthModel:
    def __init__(self):
        self.dim = Synth.synth.dim
        Synth.synth.add_model(self)

    def render(self, im, fl_i, ch_i, cy_i):
        pass


class PeaksModel(BaseSynthModel):
    def __init__(self, n_peaks=1000):
        super().__init__()
        self.n_peaks = n_peaks
        self.locs = np.zeros((n_peaks, 2))
        self.amps = np.ones((n_peaks,))

    def locs_randomize(self):
        self.locs = np.random.uniform(0, self.dim, (self.n_peaks, 2))
        return self

    def locs_randomize_no_subpixel(self):
        self.locs = np.random.uniform(0, self.dim, (self.n_peaks, 2)).astype(int)
        return self

    def locs_randomize_away_from_edges(self, dist=15):
        self.locs = np.random.uniform(
            [dist, dist], np.array(self.dim) - dist, (self.n_peaks, 2)
        )
        return self

    def locs_grid(self, pad=10):
        steps = math.floor(math.sqrt(self.n_peaks))
        self.locs = np.array(
            [
                (y, x)
                for y in np.linspace(pad, self.dim[0] - 2 * pad, steps)
                for x in np.linspace(pad, self.dim[0] - 2 * pad, steps)
            ]
        )
        return self

    def locs_add_random_subpixel(self):
        self.locs += np.random.uniform(-1, 1, self.locs.shape)
        return self

    def amps_constant(self, val):
        self.amps = val * np.ones((self.n_peaks,))
        return self

    def amps_randomize(self, mean=1000, std=10):
        self.amps = mean + std * np.random.randn(self.n_peaks)
        return self

    def dyt_amp_constant(self, amp):
        self.dyt_amp = amp
        return self

    def dyt_uniform(self, dyt):
        dyt = np.array(dyt)
        dyts = np.tile(dyt, (self.amps.shape[0], 1))
        self.amps = self.dyt_amp * dyts
        return self

    def dyt_random_choice(self, dyts, probs):
        """
        dyts is like:
            [
                [3, 2, 2, 1],
                [2, 1, 1, 0],
                [1, 0, 0, 0],
            ]

        and each row of dyts has a probability for it
            probs: [0.5, 0.3, 0.2]
        """
        dyts = np.array(dyts)
        check.array_t(dyts, ndim=2)
        assert dyts.shape[0] == len(probs)

        choices = np.random.choice(len(dyts), size=self.n_peaks, p=probs)
        self.amps = self.dyt_amp * dyts[choices, :]
        return self

    def remove_near_edges(self, dist=20):
        self.locs = np.array(
            [
                loc
                for loc in self.locs
                if dist < loc[0] < self.dim[0] - dist
                and dist < loc[1] < self.dim[1] - dist
            ]
        )
        return self


class PeaksModelPSF(PeaksModel):
    """Sample from a RegPSF"""

    def __init__(self, reg_psf: RegPSF, **kws):
        check.t(reg_psf, RegPSF)
        self.reg_psf = reg_psf
        super().__init__(**kws)

    def render(self, im, fl_i, ch_i, cy_i):
        super().render(im, fl_i, ch_i, cy_i)

        n_divs = self.reg_psf.n_divs

        for loc, amp in zip(self.locs, self.amps):
            if isinstance(amp, np.ndarray):
                amp = amp[cy_i]

            div_y, div_x = np.floor(n_divs * loc / self.reg_psf.raw_dim).astype(int)
            frac_y = np.modf(loc[0])[0]
            frac_x = np.modf(loc[1])[0]
            psf_im = self.reg_psf.render_one_reg(
                div_y, div_x, amp=amp, frac_y=frac_y, frac_x=frac_x, const=0.0
            )
            imops.accum_inplace(im, psf_im, loc=YX(*np.floor(loc)), center=True)


class PeaksModelGaussian(PeaksModel):
    def __init__(self, **kws):
        self.mea = kws.pop("mea", 11)
        super().__init__(**kws)
        self.std = None
        self.std_x = None
        self.std_y = None
        self.z_scale = None  # (simulates z stage)
        self.z_center = None  # (simulates z stage)

    def z_function(self, z_scale, z_center):
        self.z_scale = z_scale
        self.z_center = z_center
        return self

    def uniform_width_and_heights(self, width=1.5, height=1.5):
        self.std_x = [width for _ in self.locs]
        self.std_y = [height for _ in self.locs]
        return self

    def render(self, im, fl_i, ch_i, cy_i):
        if self.std_x is None:
            self.std_x = [self.std]
        if self.std_y is None:
            self.std_y = [self.std]

        n_locs = len(self.locs)
        if len(self.std_x) != n_locs:
            self.std_x = np.repeat(self.std_x, (n_locs,))
        if len(self.std_y) != n_locs:
            self.std_y = np.repeat(self.std_y, (n_locs,))

        super().render(im, fl_i, ch_i, cy_i)

        z_scale = 1.0
        if self.z_scale is not None:
            assert self.z_center is not None
            z_scale = 1.0 + self.z_scale * (cy_i - self.z_center) ** 2

        for loc, amp, std_x, std_y in zip(self.locs, self.amps, self.std_x, self.std_y):
            if isinstance(amp, np.ndarray):
                amp = amp[cy_i]

            frac_y = np.modf(loc[0])[0]
            frac_x = np.modf(loc[1])[0]
            peak_im = imops.gauss2_rho_form(
                amp=amp,
                std_x=z_scale * std_x,
                std_y=z_scale * std_y,
                pos_x=self.mea // 2 + frac_x,
                pos_y=self.mea // 2 + frac_y,
                rho=0.0,
                const=0.0,
                mea=self.mea,
            )

            imops.accum_inplace(im, peak_im, loc=YX(*np.floor(loc)), center=True)


class PeaksModelGaussianCircular(PeaksModelGaussian):
    def __init__(self, **kws):
        super().__init__(**kws)
        self.std = 1.0

    def widths_uniform(self, width=1.5):
        self.std_x = [width for _ in self.locs]
        self.std_y = copy.copy(self.std_x)
        return self

    def widths_variable(self, width=1.5, scale=0.1):
        self.std_x = [random.gauss(width, scale) for _ in self.locs]
        self.std_y = copy.copy(self.std_x)
        return self

    def render(self, im, fl_i, ch_i, cy_i):
        # self.covs = np.array([(std ** 2) * np.eye(2) for std in self.stds])
        super().render(im, fl_i, ch_i, cy_i)


class PeaksModelGaussianAstigmatism(PeaksModelGaussian):
    def __init__(self, strength, **kws):
        raise DeprecationWarning
        super().__init__(**kws)
        self.strength = strength
        center = np.array(self.dim) / 2
        d = self.dim[0]
        for loc_i, pos in enumerate(self.locs):
            delta = center - pos
            a = np.sqrt(np.sum(delta ** 2))
            r = 1 + strength * a / d
            pc0 = delta / np.sqrt(delta.dot(delta))
            pc1 = np.array([-pc0[1], pc0[0]])
            cov = np.eye(2)
            cov[0, 0] = r * pc0[1]
            cov[1, 0] = r * pc0[0]
            cov[0, 1] = pc1[1]
            cov[1, 1] = pc1[0]
            self.covs[loc_i, :, :] = cov


# Commenting this out to avoid an issue with the PSF package interacting with numpy
# it should be brought back when that issue is better understood. To duplicate the issue
# do:
#  - docker run --rm -it -v $(pwd):/erisyon/plaster jupyter/scipy-notebook:latest bash
#  - cd /erisyon/plaster && python setup.py install
#
# class PeaksModelPSF(PeaksModel):
#     def __init__(self, n_z_slices=8, depth_in_microns=0.4, r_in_microns=28.0, **kws):
#         """
#         Generates a set of psf images for each z slice called self.z_to_psf
#         The self.z_iz keeps track of which z slice each peak is assigned to.
#         """
#         super().__init__(**kws)
#         self.n_z_slices = n_z_slices
#         self.z_iz = np.zeros((self.n_peaks,), dtype=int)
#         self.z_to_psf = psf_sample(
#             n_z_slices=64, depth_in_microns=depth_in_microns, r_in_microns=r_in_microns
#         )

#     def z_randomize(self):
#         # Unrealisitically pull from any PSF z depth
#         self.z_iz = np.random.randint(0, self.n_z_slices, self.n_peaks)
#         return self

#     def z_set_all(self, z_i):
#         self.z_iz = (z_i * np.ones((self.n_peaks,))).astype(int)
#         return self

#     def render(self, im, cy_i):
#         super().render(im, cy_i)
#         for loc, amp, z_i in zip(self.locs, self.amps, self.z_iz):
#             frac_part, int_part = np.modf(loc)
#             shifted_peak_im = imops.sub_pixel_shift(self.z_to_psf[z_i], frac_part)
#             imops.accum_inplace(
#                 im, amp * shifted_peak_im, loc=YX(*int_part), center=True
#             )


class IlluminationQuadraticFalloffModel(BaseSynthModel):
    def __init__(self, center=(0.5, 0.5), width=1.2):
        super().__init__()
        self.center = center
        self.width = width

    def render(self, im, fl_i, ch_i, cy_i):
        super().render(im, fl_i, ch_i, cy_i)
        yy, xx = np.meshgrid(
            (np.linspace(0, 1, im.shape[0]) - self.center[0]) / self.width,
            (np.linspace(0, 1, im.shape[1]) - self.center[1]) / self.width,
        )
        self.regional_scale = np.exp(-(xx ** 2 + yy ** 2))
        im *= self.regional_scale


class CameraModel(BaseSynthModel):
    def __init__(self, bg_mean=100, bg_std=10):
        super().__init__()
        self.bg_mean = bg_mean
        self.bg_std = bg_std

    def render(self, im, fl_i, ch_i, cy_i):
        super().render(im, fl_i, ch_i, cy_i)
        bg = np.random.normal(loc=self.bg_mean, scale=self.bg_std, size=self.dim)
        imops.accum_inplace(im, bg, XY(0, 0), center=False)


class HaloModel(BaseSynthModel):
    def __init__(self, std=20, scale=2):
        super().__init__()
        self.std = std
        self.scale = scale

    def render(self, im, fl_i, ch_i, cy_i):
        super().render(im, fl_i, ch_i, cy_i)
        size = int(self.std * 2.5)
        size += 1 if size % 2 == 0 else 0
        bg_mean = np.median(im) - 1
        blur = cv2.GaussianBlur(im, (size, size), self.std) - bg_mean - 1
        imops.accum_inplace(im, self.scale * blur, XY(0, 0), center=False)


def synth_to_ims_import_result(synth: Synth):
    chcy_ims = synth.render_chcy()

    ims_import_params = ImsImportParams()
    ims_import_result = ImsImportResult(
        params=ims_import_params,
        tsv_data=None,
        n_fields=synth.n_fields,
        n_channels=synth.n_channels,
        n_cycles=synth.n_cycles,
        dim=synth.dim[0],
        dtype=np.dtype(OUTPUT_NP_TYPE).name,
        src_dir="",
    )

    with tmp_folder(remove=False):
        for fl_i in range(synth.n_fields):
            field_chcy_arr = ims_import_result.allocate_field(
                fl_i,
                (synth.n_channels, synth.n_cycles, synth.dim[0], synth.dim[1]),
                OUTPUT_NP_TYPE,
            )
            field_chcy_ims = field_chcy_arr.arr()

            field_chcy_ims[:, :, :, :] = chcy_ims

            ims_import_result.save_field(fl_i, field_chcy_arr, None, None)

        ims_import_result.save()

    return ims_import_result
