import numpy as np
import pickle
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.sigproc_v2_task import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.image import imops
from plaster.tools.image.coord import ROI

# from plaster.run.sigproc_v2.psf_sample import psf_sample
from plaster.tools.log.log import debug
from plaster.tools.schema.check import CheckAffirmError
from plaster.tools.utils import utils
from plaster.tools.utils.tmp import tmp_folder, tmp_file
from zest import zest


def zest_kernel():
    def it_has_zero_mean():
        kern = worker._kernel()
        assert np.isclose(np.mean(kern), 0.0)

    zest()


def zest_intersection_roi_from_aln_offsets():
    def it_raises_on_first_not_0_0():
        with zest.raises(CheckAffirmError, in_message="(0,0)"):
            worker._intersection_roi_from_aln_offsets([(1, 0), (1, 0)], (10, 10))

    def it_returns_the_intersection():
        roi = worker._intersection_roi_from_aln_offsets(
            [(0, 0), (1, 0), (0, 1)], (10, 10)
        )
        assert roi == ROI((0, 0), (9, 9))

    def it_returns_an_empty_ROI():
        roi = worker._intersection_roi_from_aln_offsets(
            [(0, 0), (5, 0), (-5, 0)], (10, 10)
        )
        assert roi[0].stop - roi[0].start == 0

    def it_bounds():
        offsets = np.array(
            [
                [0, 0],
                [2, -2],
                [-8, 1],
                [-15, 2],
                [-8, -3],
                [-8, -2],
                [-8, -1],
                [-17, -5],
                [-8, -11],
            ],
        )
        raw_dim = (1024, 1024)
        roi = worker._intersection_roi_from_aln_offsets(offsets, raw_dim)
        expected = ROI((17, 11), (1024 - 17 - 2, 1024 - 11 - 2))
        assert roi == expected

    def it_bounds_1():
        offsets = np.array(
            [
                [0, 0],
                [-5, -2],
                [-8, 1],
                [-15, 2],
                [-8, -3],
                [-8, -2],
                [-8, -1],
                [-17, -5],
                [-8, -11],
            ],
        )
        raw_dim = (1024, 1024)
        roi = worker._intersection_roi_from_aln_offsets(offsets, raw_dim)
        expected = ROI((17, 11), (1024 - 17, 1024 - 11 - 2))
        assert roi == expected

    zest()


def zest_regional_balance_chcy_ims():
    def _setup(corner_bal):
        divs = 5
        bg = 100 * np.ones((divs, divs))
        bal = np.ones((divs, divs))
        bal[0, 0] = corner_bal

        chcy_ims = 101 * np.ones((2, 4, 512, 512))
        chcy_ims[1, :, :, :] += 1

        calib = Calibration(
            {
                "regional_bg_mean.instrument_channel[0]": bg.tolist(),
                "regional_illumination_balance.instrument_channel[0]": bal.tolist(),
                "regional_bg_mean.instrument_channel[1]": bg.tolist(),
                "regional_illumination_balance.instrument_channel[1]": bal.tolist(),
            }
        )

        return chcy_ims, calib

    def it_subtracts_regional_bg():
        chcy_ims, calib = _setup(10.0)
        bal_ims = worker._regional_balance_chcy_ims(chcy_ims, calib)

        # 0, 0 corner is 10 times brighter
        # Opposite corner is just unit brightness
        assert np.all(np.abs(bal_ims[0, :, 0, 0] - (101 - 100) * 10) < 1.0)
        assert np.all(np.abs(bal_ims[0, :, -1, -1] - (101 - 100) * 1) < 1.0)
        assert np.all(np.abs(bal_ims[1, :, 0, 0] - (102 - 100) * 10) < 1.0)
        assert np.all(np.abs(bal_ims[1, :, -1, -1] - (102 - 100) * 1) < 1.0)

    def it_handles_all_zeros():
        _, calib = _setup(1.0)
        all_zeros = np.zeros((2, 4, 512, 512))
        bal_ims = worker._regional_balance_chcy_ims(all_zeros, calib)
        assert np.all(np.abs(bal_ims - (0 - 100) * 1) < 1.0)

    def it_raises_on_nans():
        chcy_ims, calib = _setup(1.0)

        with zest.raises(ValueError, in_args="nan"):
            nan_chcy_ims = np.full_like(chcy_ims, np.nan)
            worker._regional_balance_chcy_ims(nan_chcy_ims, calib)

        with zest.raises(ValueError, in_args="nan"):
            calib[f"regional_bg_mean.instrument_channel[0]"][0][0] = np.nan
            worker._regional_balance_chcy_ims(chcy_ims, calib)

    zest()


def zest_peak_find():
    bg_mean = 145

    def it_finds_peaks_as_they_approach():
        for dist in np.linspace(10, 0, 10):
            with synth.Synth(overwrite=True, dim=(128, 128)) as s:
                peaks = (
                    synth.PeaksModelGaussianCircular(n_peaks=2)
                    .amps_constant(val=4_000)
                    .widths_uniform(1.5)
                )
                synth.CameraModel(bias=bg_mean, std=11)
                peaks.locs[0] = (64, 64 - dist)
                peaks.locs[1] = (64, 64 + dist)
                im = s.render_chcy()[0, 0]
                im = im - bg_mean
            locs = worker._peak_find(im)
            if dist > 1.5:
                # After 1.5 pixels (total of 3 pixels) then they peaks should merge
                assert locs.shape == (2, 2)
            assert np.all((locs[:, 0] - 64) ** 2 < 1.5 ** 2)

    def it_finds_peaks_as_density_increases():
        _expected = [
            [100, 92, 9],  # n_spots to generate, n expected to find, within bounds
            [125, 113, 10],
            [150, 135, 15],
            [175, 146, 20],
            [200, 175, 25],
        ]
        for expected, n_peaks in zip(_expected, np.linspace(100, 200, 5).astype(int)):
            with synth.Synth(overwrite=True, dim=(256, 256)) as s:
                peaks = (
                    synth.PeaksModelGaussianCircular(n_peaks=n_peaks)
                    .amps_constant(val=4_000)
                    .widths_uniform(1.5)
                    .locs_randomize_away_from_edges()
                )
                synth.CameraModel(bias=bg_mean, std=11)
                im = s.render_chcy()[0, 0]
                im = im - bg_mean

            locs = worker._peak_find(im)
            assert expected[0] == n_peaks
            if not utils.np_within(locs.shape[0], expected[1], expected[2]):
                raise AssertionError(
                    f"density increase did not find expected peaks. "
                    f"Found: {locs.shape[0]}, Expected {expected[1]} +/- {expected[2]}"
                )

    zest()


@zest.group("slow")
def zest_psf_estimate():
    bg_mean = 145
    bg_std = 11

    def _make_image(depth, n_peaks=1000):
        # This table is based on a fitting of the PSF using polymer spheres
        # s3://erisyon-acquire/Val/ESN/2020/2020_05/ESN_2020_05_21_amb/ESN_2020_21_amb_02_NSdeepredZ
        # Using the exploration notebook internal/explore/sigproc/psf_with_real_data.ipynb
        # The depth is set so that 0 microns is optimally in focus
        z_microns_to_peak_std = np.array(
            [
                [-0.25, 2.187504401090129],
                [-0.2, 2.063845408774231],
                [-0.15000000000000002, 2.001985762818282],
                [-0.1, 1.9433957713576882],
                [-0.05, 1.9091776019044606],
                [0.0, 1.8891420470361429],
                [0.05, 1.8951213618125622],
                [0.1, 1.9476507707766804],
                [0.15000000000000002, 2.0169404372571758],
                [0.2, 2.091394093944999],
                [0.25, 2.4354449946062364],
            ]
        )

        # Background parameters based on:
        # s3://erisyon-acquire/Val/ESN/2020/2020_05/ESN_2020_05_21_amb/ESN_2020_21_amb_01_JSP092Z
        with synth.Synth(overwrite=True, dim=(1024, 1024)) as s:
            idx = utils.np_arg_find_nearest(z_microns_to_peak_std[:, 0], depth)
            std = z_microns_to_peak_std[idx, 1]
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=n_peaks)
                .locs_randomize()
                .amps_constant(val=4_000)
                # Amp of 4000 based on fitting of a peptide from: erisyon-acquire/Val/ESN/2020/2020_05/ESN_2020_05_21_amb/ESN_2020_21_amb_01_JSP092Z
                .widths_uniform(std)
            )
            synth.CameraModel(bias=bg_mean, std=bg_std)
            im = s.render_chcy()[0, 0]
            im = im - bg_mean
        return peaks, im, std

    def _make_image_n_locs(locs):
        with synth.Synth(overwrite=True, dim=(128, 128)) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=len(locs))
                .amps_constant(val=4_000)
                .widths_uniform(1.5)
            )
            synth.CameraModel(bias=bg_mean, std=bg_std)
            peaks.locs = locs
            im = s.render_chcy()[0, 0]
            im = im - bg_mean
        return peaks, im

    def it_extracts_a_clean_psf_with_subpixel_alignment():
        # Sweep over various psf z depths
        # Generate a seris of small synth images (representing a region over many fields)
        # Pass in the locs directly from the synth image maker (bypassing peak finder)
        # Check that we get back a good approximation of the PSF

        for i, depth in enumerate(np.linspace(-0.25, 0.25, 4)):
            peaks, im, expected_std = _make_image(depth)

            psf, reasons = worker._psf_estimate(
                im, peaks.locs, mea=17, return_reasons=True
            )

            fit_params, _ = imops.fit_gauss2(psf)
            got = np.array(fit_params)
            expected = np.array([np.nan, expected_std, expected_std, 8, 8, 0, 0, 17])
            assert np.all((got[1:] - expected[1:]) ** 2 < 0.15 ** 2)

    def it_skips_near_edges():
        peaks, im, std = _make_image(0.0)
        psf, reasons = worker._psf_estimate(im, peaks.locs, mea=17, return_reasons=True)
        for loc, reason in zip(peaks.locs, reasons):
            skipped = reason[worker.PSFEstimateMaskFields.skipped_near_edges]
            if loc[0] < 5 or loc[0] > 1024 - 5 or loc[1] < 5 or loc[1] > 1024 - 5:
                assert skipped == 1

    def it_skips_contentions():
        peaks, im = _make_image_n_locs([[64, 62], [64, 66]])
        psf, reasons = worker._psf_estimate(im, peaks.locs, mea=17, return_reasons=True)
        for loc, reason in zip(peaks.locs, reasons):
            assert reason[worker.PSFEstimateMaskFields.skipped_too_crowded] == 1

    def it_does_not_skip_contentions():
        peaks, im = _make_image_n_locs([[64, 50], [64, 60]])
        psf, reasons = worker._psf_estimate(im, peaks.locs, mea=17, return_reasons=True)
        for loc, reason in zip(peaks.locs, reasons):
            assert reason[worker.PSFEstimateMaskFields.skipped_too_crowded] == 0

    def it_skips_nans():
        peaks, im = _make_image_n_locs([[64, 50], [64, 70]])
        im[64, 50] = np.nan
        psf, reasons = worker._psf_estimate(im, peaks.locs, mea=17, return_reasons=True)
        for loc, reason in zip(peaks.locs, reasons):
            if loc[1] < 64:
                assert reason[worker.PSFEstimateMaskFields.skipped_has_nan] == 1
            else:
                assert reason[worker.PSFEstimateMaskFields.skipped_has_nan] == 0

    def it_skips_darks():
        peaks, im = _make_image_n_locs([[64, 54]])
        psf, reasons = worker._psf_estimate(
            im, peaks.locs, mea=17, threshold_abs=bg_std * 3, return_reasons=True
        )
        peaks.locs = [[64, 54], [64, 80]]
        for loc, reason in zip(peaks.locs, reasons):
            if loc[1] < 64:
                assert reason[worker.PSFEstimateMaskFields.skipped_too_dark] == 0
            else:
                assert reason[worker.PSFEstimateMaskFields.skipped_too_dark] == 1

    def it_skips_too_oval():
        """It doesn't have to work perfectly every time, but pass once in a 10 try loop"""
        n_fails = 0
        for i in range(10):
            failed = 0
            locs = [[64, 54], [64, 70]]
            with synth.Synth(overwrite=True, dim=(128, 128)) as s:
                peaks = synth.PeaksModelGaussianCircular(
                    n_peaks=len(locs)
                ).amps_constant(val=4_000)
                peaks.std_x = [1.0, 1.0]
                peaks.std_y = [1.0, 2.0]
                synth.CameraModel(bias=bg_mean, std=bg_std)
                peaks.locs = locs
                im = s.render_chcy()[0, 0]
                im = im - bg_mean

            psf, reasons = worker._psf_estimate(
                im, peaks.locs, mea=17, return_reasons=True
            )
            for loc, reason in zip(peaks.locs, reasons):
                if loc[1] < 64:
                    if reason[worker.PSFEstimateMaskFields.skipped_too_oval] != 0:
                        failed += 1
                else:
                    if reason[worker.PSFEstimateMaskFields.skipped_too_oval] != 1:
                        failed += 1

            if failed != 0:
                n_fails += 1

        assert n_fails < 3

    def it_normalizes():
        peaks, im, std = _make_image(0.0, 100)
        psf = worker._psf_estimate(im, peaks.locs, mea=17, return_reasons=False)
        assert utils.np_within(np.sum(psf), 1.0, 0.001)

    def it_returns_reason_by_default():
        im = np.zeros((100, 100))
        psf, reasons = worker._psf_estimate(im, [[64, 64]], mea=17)
        assert psf.shape == (17, 17) and reasons.shape == (1, 8)

    def it_does_not_return_reasons_if_requested():
        im = np.zeros((100, 100))
        psf = worker._psf_estimate(im, [[64, 64]], mea=17, return_reasons=False)
        assert psf.shape == (17, 17)

    zest()


"""
def zest_psf_normalize():
    def it_normalizes_4_dim():
        psfs = np.ones((2, 2, 4, 4))
        got = worker._psf_normalize(psfs)
        assert got.shape == (2, 2, 4, 4) and np.all(got == 1.0 / 16.0)

    def it_normalizes_5_dim():
        psfs = np.ones((3, 2, 2, 4, 4))
        psfs[0] = psfs[0] * 1
        psfs[1] = psfs[1] * 2
        psfs[2] = psfs[2] * 3
        got = worker._psf_normalize(psfs)
        assert got.shape == (3, 2, 2, 4, 4) and np.all(got == 1.0 / 16.0)

    def it_handles_zeros():
        psfs = np.zeros((2, 2, 4, 4))
        got = worker._psf_normalize(psfs)
        assert np.all(got == 0.0)

    zest()
"""


def zest_calibrate_bg_and_psf_im():
    def it_gets_bg_mean_and_std():
        im = np.random.normal(loc=100, scale=10, size=(256, 256))
        reg_bg_mean, reg_bg_std = worker._background_estimate_im(im, divs=5)
        assert np.all(100.0 - reg_bg_mean < 4.0 ** 2)
        assert np.all(10.0 - reg_bg_std < 1.0 ** 2)

    zest()


"""
There is a lot of complexity here regarding channel ordering.

I've written the calibration so that the channels are numbered
and there is no remapping.

But the original form of this allowed a remapping "atto=1" where
atto is an output channel and 1 is the input.

In reality, each device will have channel numbers that have specific filters
and there will not be any remapping.

So, this is extremely confusing right now and needs to be simplified.
TODO Talk to Angela

For now I'm keeping the input order the same as the output

"""


@zest.skip(reason="un-skip once we have multi-channel working")
def zest_compute_channel_weights():
    def it_returns_balanced_channels():
        with tmp_file() as cal_file:
            calibration = Calibration(
                {
                    "regional_bg_mean.instrument_channel[0].test": [
                        [100.0, 100.0],
                        [100.0, 100.0],
                    ],
                    "regional_bg_mean.instrument_channel[1].test": [
                        [200.0, 200.0],
                        [200.0, 200.0],
                    ],
                }
            )
            pickle.dump(calibration, open(cal_file, "wb"))
            sigproc_params = SigprocV2Params(
                mode="analyze",
                radiometry_channels=dict(aaa=0, bbb=1),
                calibration_file=cal_file,
                instrument_subject_id="test",
            )

            balance = worker._analyze_step_1a_compute_channel_weights(sigproc_params)
            assert np.all(balance == [2.0, 1.0])

    zest()


def zest_import_balanced_images():
    @zest.skip(reason="un-skip when we  have multi-channel working")
    def it_remaps_and_balances_channels():
        with tmp_file() as cal_file:
            calibration = Calibration(
                {
                    "regional_illumination_balance.instrument_channel[0].test": [
                        [1.0, 1.0],
                        [1.0, 1.0],
                    ],
                    "regional_illumination_balance.instrument_channel[1].test": [
                        [1.0, 1.0],
                        [1.0, 1.0],
                    ],
                    "regional_bg_mean.instrument_channel[0].test": [
                        [100.0, 100.0],
                        [100.0, 100.0],
                    ],
                    "regional_bg_mean.instrument_channel[1].test": [
                        [200.0, 200.0],
                        [200.0, 200.0],
                    ],
                }
            )
            pickle.dump(calibration, open(cal_file, "wb"))
            sigproc_params = SigprocV2Params(
                mode="analyze",
                radiometry_channels=dict(aaa=0, bbb=1),
                calibration_file=cal_file,
                instrument_subject_id="test",
            )
            chcy_ims = np.ones((2, 1, 128, 128))
            chcy_ims[0] *= 1000.0
            chcy_ims[1] *= 2000.0
            balanced_ims = worker._analyze_step_1_import_balanced_images(
                chcy_ims, sigproc_params, calibration
            )
            assert np.all(np.isclose(balanced_ims[0], (1000 - 100) * 2))
            assert np.all(np.isclose(balanced_ims[1], (2000 - 200) * 1))

    def it_balances_regionally():
        with tmp_file() as cal_file:
            calibration = Calibration(
                {
                    "regional_illumination_balance.instrument_channel[0]": [
                        [1.0, 5.0],
                        [1.0, 1.0],
                    ],
                    # "regional_illumination_balance.instrument_channel[1]": [
                    #     [7.0, 1.0],
                    #     [1.0, 1.0],
                    # ],
                    "regional_bg_mean.instrument_channel[0]": [
                        [100.0, 100.0],
                        [100.0, 100.0],
                    ],
                    # "regional_bg_mean.instrument_channel[1]": [
                    #     [200.0, 200.0],
                    #     [200.0, 200.0],
                    # ],
                }
            )
            pickle.dump(calibration, open(cal_file, "wb"))
            sigproc_params = SigprocV2Params(mode="analyze", calibration_file=cal_file,)
            chcy_ims = np.ones((1, 1, 128, 128))
            # chcy_ims = np.ones((2, 1, 128, 128))
            chcy_ims[0] *= 1000.0
            # chcy_ims[1] *= 2000.0
            balanced_ims = worker._analyze_step_1_import_balanced_images(
                chcy_ims, sigproc_params, calibration
            )
            assert np.all(np.isclose(balanced_ims[0, 0, 0, 0], (1000 - 100)))
            assert np.all(np.isclose(balanced_ims[0, 0, 0, 127], (1000 - 100) * 5))
            # assert np.all(np.isclose(balanced_ims[1, 0, 0, 0], (2000 - 200) * 1 * 7))
            # assert np.all(np.isclose(balanced_ims[1, 0, 127, 0], (2000 - 200) * 1))

    zest()


def zest_mask_anomalies_im():
    def _im():
        bg_mean = 145
        with synth.Synth(overwrite=True, dim=(512, 512)) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=1000)
                .amps_constant(val=4_000)
                .locs_randomize()
            )
            synth.CameraModel(bias=bg_mean, std=14)
            im = s.render_chcy()[0, 0]
            im = im - bg_mean
            return im

    def it_does_not_mask_point_anomalies():
        im = _im()
        im[5:10, 5:10] = np.random.normal(loc=4_000, scale=20, size=(5, 5))
        res = worker._analyze_step_2_mask_anomalies_im(im, den_threshold=300)
        n_nan = np.sum(np.isnan(res))
        frac_nan = n_nan / (res.shape[0] * res.shape[1])
        assert frac_nan < 0.001

    def it_masks_large_anomalies():
        im = _im()
        im[50:80, 50:80] = np.random.normal(loc=4_000, scale=20, size=(30, 30))
        res = worker._analyze_step_2_mask_anomalies_im(im, den_threshold=300)
        assert np.all(np.isnan(res[50:80, 50:80]))

        # Clear out the nan area (and some extra)
        # and allow for 1% of the remainder to be nan
        res[40:90, 40:90] = 0.0
        n_nan = np.sum(np.isnan(res))
        frac_nan = n_nan / (res.shape[0] * res.shape[1])
        assert frac_nan < 0.01

    zest()


@zest.group("slow")
def zest_align():
    def _ims(mea=512, std=1.5):
        bg_mean = 145
        with synth.Synth(n_cycles=3, overwrite=True, dim=(mea, mea)) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=1000)
                .amps_constant(val=4_000)
                .locs_randomize()
                .widths_uniform(std)
            )
            synth.CameraModel(bias=bg_mean, std=14)
            cy_ims = s.render_chcy()[0]
            return cy_ims, s.aln_offsets

    def it_removes_the_noise_floor():
        cy_ims, true_aln_offsets = _ims()
        pred_aln_offsets, aln_scores = worker._analyze_step_3_align(cy_ims)
        assert np.all(true_aln_offsets == pred_aln_offsets)

    def it_is_robust_to_different_image_sizes():
        cy_ims, true_aln_offsets = _ims(mea=128)
        pred_aln_offsets, aln_scores = worker._analyze_step_3_align(cy_ims)
        assert np.all(true_aln_offsets == pred_aln_offsets)

    def it_is_robust_to_different_peak_sizes():
        cy_ims, true_aln_offsets = _ims(std=3.0)
        pred_aln_offsets, aln_scores = worker._analyze_step_3_align(cy_ims)
        assert np.all(true_aln_offsets == pred_aln_offsets)

    zest()


@zest.group("slow")
def zest_composite_with_alignment_offsets_chcy_ims():
    def _ims():
        bg_mean = 145
        with synth.Synth(n_channels=2, n_cycles=3, overwrite=True, dim=(256, 256)) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=1000)
                .amps_constant(val=4_000)
                .locs_randomize()
                .widths_uniform(1.5)
            )
            synth.CameraModel(bias=bg_mean, std=14)
            chcy_ims = s.render_chcy()
            return chcy_ims, s.aln_offsets

    def it_creates_a_single_composite_image():
        chcy_ims, true_aln_offsets = _ims()
        comp_im = worker._analyze_step_4_composite_with_alignment_offsets_chcy_ims(
            chcy_ims, true_aln_offsets
        )
        assert comp_im.ndim == 4
        assert (
            comp_im.shape[0] == chcy_ims.shape[0]
            and comp_im.shape[1] == chcy_ims.shape[1]
        )
        assert (
            comp_im.shape[2] < chcy_ims.shape[2] or comp_im.shape[3] < chcy_ims.shape[3]
        )
        for cy in range(2):
            for ch in range(2):
                diff = np.sum(comp_im[ch, cy + 1, :, :] - comp_im[ch, cy, :, :])
                if not utils.np_within(diff, 0.0, 14_000):
                    debug(diff)
                assert utils.np_within(diff, 0.0, 14_000)

    zest()


def zest_peak_radiometry():
    # TODO: Decide about the center weighing mask. Is it helping or hurting? See next

    def _im(off, bg_std=14.0):
        bg_mean = 145
        mea = 23

        psf = imops.gauss2_rho_form(
            amp=1.0,
            std_x=1.5,
            std_y=1.5,
            pos_x=mea // 2,
            pos_y=mea // 2,
            rho=0.0,
            const=0.0,
            mea=mea,
        )

        with synth.Synth(overwrite=True, dim=(mea, mea)) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=1)
                .amps_constant(val=1_000)
                .widths_uniform(1.5)
            )
            peaks.locs = [(mea // 2 + off[0], mea // 2 + off[1])]
            synth.CameraModel(bias=bg_mean, std=bg_std)
            im = s.render_chcy()[0, 0] - bg_mean
            return im, psf

    def it_gets_a_perfect_result_with_no_noise_and_perfect_alignment():
        im, psf = _im(off=(0.0, 0.0), bg_std=0.0)
        signal, noise = worker._analyze_step_6a_peak_radiometry(
            im, psf, np.ones_like(im)
        )
        assert np.isclose(signal, 1_000)

    def it_sub_pixel_alignments_no_noise():
        for x in np.linspace(-0.5, 0.5, 11):
            signals = []
            for trials in range(10):
                im, psf = _im(off=(0.0, x), bg_std=0.0)
                signal, noise = worker._analyze_step_6a_peak_radiometry(
                    im, psf, np.ones_like(im)
                )
                signals += [signal]
            assert 999.0 <= np.mean(signals) <= 1010.0

    def it_gets_the_residuals():
        signals = []
        noises = []
        for trials in range(100):
            im, psf = _im(off=(0.0, 0.0), bg_std=10.0)
            signal, noise = worker._analyze_step_6a_peak_radiometry(
                im, psf, np.ones_like(im)
            )
            signals += [signal]
            noises += [noise]
        assert 950.0 < np.mean(signals) < 1050.0
        assert 45.0 < np.mean(noises) < 55.0

    def it_nans_negative_signal_or_noise():
        im, psf = _im(off=(100.0, 0.0), bg_std=10.0)
        im = np.full_like(im, -100.0)
        signal, noise = worker._analyze_step_6a_peak_radiometry(
            im, psf, np.ones_like(im)
        )
        assert np.isnan(signal) and np.isnan(noise)

    zest()


# def zest_radiometry():
#     def it_uses_the_regional_psf():
#         raise NotImplementedError
#
#     def it_skips_near_edges():
#         raise NotImplementedError
#
#     def it_skips_nans():
#         # How does this ever happen? Can it happen?
#         raise NotImplementedError
#
#     def it_calls_to_peak_radiometry():
#         raise NotImplementedError
#
#     def it_returns_sig_and_noise_by_loc_ch_cy():
#         raise NotImplementedError
#
#     zest()
#
#
# def zest_sigproc_field():
#     def it_works_on_a_test_field():
#         # Is this going to characterize? Or random seed?
#         raise NotImplementedError
#
#     def it_filters_empties():
#         # Maybe this should be its own unit?
#         raise NotImplementedError
#
#     def it_returns_chcy_ims__locs__radmat__aln_offsets__aln_scores():
#         raise NotImplementedError
#
#     zest()
#
# def zest_sigproc():
#     def it_works_on_two_fields():
#         raise NotImplementedError
#
#     zest()


# # Helpers
# DONE def _kernel():
# DONE def _intersection_roi_from_aln_offsets(aln_offsets, raw_dim):
# DONE def _regional_bg_fg_stats(im, mask_radius=2, divs=5, return_ims=False):
# DONE def _regional_balance_chcy_ims(chcy_ims, calib):
# SKIP   def circle_locs(im, locs, inner_radius=3, outer_radius=4, fill_mode="nan"):
# DONE def _peak_find(im):
#
# # PSF
# DONE def _psf_estimate(im, locs, mea, keep_dist=8, return_reasons=True):
# DONE def _psf_normalize(psfs):
#
# # Calibration
# def _calibrate_bg_and_psf_im(im, divs=5, keep_dist=8, peak_mea=11, locs=None):
# def _calibrate(flchcy_ims, divs=5, progress=None, overload_psf=None):
# def calibrate(ims_import_res, n_best_fields=6, divs=5, metadata=None, progress=None):
#
# # Steps
# DONE def _compute_channel_weights(sigproc_params, calib):
# DONE def _import_balanced_images(chcy_ims, sigproc_params, calib):
# DONE def _mask_anomalies_im(im, den_threshold=300):
# DONE def _align(cy_ims):
# DONE def _composite_with_alignment_offsets_chcy_ims(chcy_ims, aln_offsets):
# DONE def _peak_radiometry( peak_im, psf_kernel, center_weighted_mask, allow_non_unity_psf_kernel=False):
# def _radiometry(chcy_ims, locs, ch_z_reg_psfs, cycle_to_z_index):
#
# # Entrypoint
# def sigproc_field(chcy_ims, sigproc_params, snr_thresh=None):
# def _do_sigproc_field(ims_import_result, sigproc_params, field_i, sigproc_result):
# def sigproc(sigproc_params, ims_import_result, progress=None):
