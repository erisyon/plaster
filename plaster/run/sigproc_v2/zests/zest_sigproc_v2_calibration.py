from math import floor, ceil

import numpy as np
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.image import imops
from plaster.tools.log.log import debug
from plaster.tools.utils.utils import np_within
from plaster.tools.schema import check

from zest import zest

from munch import Munch
import numpy as np
from plumbum import local
from plaster.run.base_result import BaseResult, ArrayResult
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.tools.utils.fancy_indexer import FancyIndexer
from plaster.tools.log.log import debug


class MockImsImportResult:
    @property
    def ims(self):
        return self.ims_to_return

    def n_fields_channel_cycles(self):
        return self.n_fields, self.n_channels, self.n_cycles

    def __init__(self, ims_to_return, n_fields, n_channels, n_cycles):
        self.ims_to_return = ims_to_return
        self.n_fields = n_fields
        self.n_channels = n_channels
        self.n_cycles = n_cycles
        self.params = ImsImportParams(is_movie=True,)


def result_from_z_stack(n_fields=1, n_channels=1, n_cycles=1, uniformity="uniform"):
    # use synth to make imgs
    s = synth.Synth(
        n_fields=n_fields, n_channels=n_channels, n_cycles=n_cycles, overwrite=True
    )
    peaks = (
        synth.PeaksModelGaussianCircular(n_peaks=800)
        .locs_randomize()
        .amps_constant(val=10000)
    )
    synth.CameraModel(bias=100, std=10)

    psf_std = 0.5
    peaks.widths_uniform(psf_std)
    imgs = s.render_flchcy()
    # stuff imgs into instantiated MockImsImportResult
    mock_ims_import_result = MockImsImportResult(
        ims_to_return=imgs, n_fields=n_fields, n_channels=n_channels, n_cycles=n_cycles
    )
    return mock_ims_import_result


def grid_walk(divs):
    for x in range(0, divs):
        for y in range(0, divs):
            yield x, y


def zest_sigproc_v2_calibration():
    divs = None
    tgt_mean = None
    tgt_std = None
    peak_mea = None
    peak_dim = None
    midpt = None
    true_params = {}  # if None pylint gets confused

    def _before():
        nonlocal divs, tgt_mean, tgt_std, peak_mea, midpt, true_params, peak_dim
        divs = 5
        tgt_mean = 100
        tgt_std = 10
        peak_mea = 11
        peak_dim = (peak_mea, peak_mea)
        midpt = 5
        # true values, and allowed range, for parms returned by imops
        # fit_gauss2(): amp, std_x, std_y, pos_x, pos_y, rho, const, mea
        true_params = {
            "amp": {"tgt": 1, "range": 0.1},
            "std_x": {"tgt": None, "range": 0.15},
            "std_y": {"tgt": None, "range": 0.15},
            "pos_x": {"tgt": midpt, "range": 0.10},
            "pos_y": {"tgt": midpt, "range": 0.10},
            "rho": {"tgt": 0, "range": 0.07},
            "const": {"tgt": 0, "range": 0.05},
            "mea": {"tgt": peak_mea, "range": 0.1},
        }

    def _compare_fit_params(true_params, fit_params):
        for ix, parm in enumerate(
            ["amp", "std_x", "std_y", "pos_x", "pos_y", "rho", "const", "mea",]
        ):
            try:
                assert true_params[parm]["range"] > abs(
                    true_params[parm]["tgt"] - fit_params[ix]
                )
            except AssertionError:
                tgt = true_params[parm]["tgt"]
                actual = fit_params[ix]
                range = true_params[parm]["range"]
                debug_statement = f"{parm} {tgt} {actual} {range}"
                debug(debug_statement)
                raise

    def it_estimates_uniform_background_correctly():
        tgt_mean = 200
        tgt_std = 15
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            im = s.render_chcy()[0, 0]
        bg_mean, bg_std = worker._background_estimate(im, divs)
        assert np_within(np.mean(bg_mean), tgt_mean, 1)
        assert np_within(np.mean(bg_std), tgt_std, 1)

    def it_subtracts_uniform_bg_mean_correctly():
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            im = s.render_chcy()[0, 0]
        bg_mean, bg_std = worker._background_estimate(im, divs)
        im_sub = worker._background_subtraction(im, bg_mean)
        new_mean, new_std = worker._background_estimate(im_sub, divs)
        assert np_within(np.mean(new_mean), 0, (1 / tgt_std))

    def it_adds_regional_bg_stats_to_calib_correctly():
        calib = Calibration()
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            ims = s.render_flchcy()
        calib = worker.add_regional_bg_stats_to_calib(ims, 0, 1, divs, calib)
        bg_mean = np.array(calib["regional_bg_mean.instrument_channel[0]"])
        assert len(bg_mean.shape) == 2
        assert (bg_mean > tgt_mean - 1).all()
        assert (bg_mean < tgt_mean + 1).all()
        bg_std = np.array(calib["regional_bg_std.instrument_channel[0]"])
        assert len(bg_std.shape) == 2
        assert (bg_std > tgt_std - 1).all()
        assert (bg_std < tgt_std + 1).all()

    def it_estimates_nonuniform_bg_mean_correctly():
        tgt_mean = 200
        tgt_std = 15
        # make an image...
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            im = s.render_chcy()[0, 0]
        # make the outside border half as light, width of one div
        border = int(im.shape[0] / divs)
        for x in range(0, im.shape[0]):
            for y in range(0, im.shape[1]):
                dist_to_edge = min(x, im.shape[0] - x, y, im.shape[1] - y)
                if dist_to_edge < border:
                    im[x][y] *= 0.5
        # find mean and std, check that it got nonuniformity of means
        bg_mean, bg_std = worker._background_estimate(im, divs)
        for x, y in grid_walk(divs):
            if x in [0, divs - 1] or y in [0, divs - 1]:
                assert np_within((tgt_mean * 0.5), bg_mean[x][y], tgt_std)
                assert np_within(tgt_std * 0.5, bg_std[x][y], tgt_std / 3)
            else:
                assert np_within(tgt_mean, bg_mean[x][y], tgt_std)
                assert np_within(tgt_std, bg_std[x][y], tgt_std / 3)

    def it_estimates_nonuniform_bg_std_correctly():
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            im1 = s.render_chcy()[0][0]
        # create second image with larger std, to use for border region
        tgt_std2 = 30
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std2)
            im2 = s.render_chcy()[0][0]
        # merge the two images into one with a nonuniform std
        im3 = im1.copy()
        border = int(im3.shape[0] / divs)
        for x in range(0, im3.shape[0]):
            for y in range(0, im3.shape[1]):
                dist_to_edge = min(x, im3.shape[0] - x, y, im3.shape[1] - y)
                if dist_to_edge < border:
                    im3[x][y] = im2[x][y]
        # find mean and std, check that it detected nonuniformity of stds
        bg_mean, bg_std = worker._background_estimate(im3, divs)
        for x in range(0, divs):
            for y in range(0, divs):
                if x in [0, divs - 1] or y in [0, divs - 1]:
                    assert np_within(bg_std[x][y], tgt_std2, tgt_std2 / 2)
                    assert np_within(bg_mean[x][y], tgt_mean, tgt_std2)
                else:
                    assert np_within(bg_std[x][y], tgt_std, tgt_std / 2)
                    assert np_within(bg_mean[x][y], tgt_mean, tgt_std)
        return True

    def it_subtracts_nonuniform_bg_mean_correctly():
        divs = 15
        border_ratio = 5
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            im = s.render_chcy()[0, 0]
        # make the outside 1/border_ratio thickness darker,
        # down to 1/2 at very edge
        border = int(im.shape[0] / border_ratio)
        for x in range(0, im.shape[0]):
            for y in range(0, im.shape[1]):
                dist_to_edge = min(x, im.shape[0] - x, y, im.shape[1] - y)
                if dist_to_edge < border:
                    im[x][y] *= 0.5 + (dist_to_edge / (2 * border))
        bg_mean, bg_std = worker._background_estimate(im, divs)
        im_sub = worker._background_subtraction(im, bg_mean)
        new_mean, new_std = worker._background_estimate(im_sub, divs)
        assert np_within(np.mean(new_mean), 0, tgt_std)
        return True

    def it_can_calibrate_psf_uniform_im():
        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=400)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=tgt_mean, std=tgt_std)

        psf_std = 0.5
        peaks.widths_uniform(psf_std)
        imgs = s.render_chcy()
        bg_mean, bg_std = worker._background_estimate(imgs[0, 0], divs)
        im_sub = worker._background_subtraction(imgs[0, 0], bg_mean)
        locs, reg_psfs = worker._calibrate_psf_im(im_sub, divs=divs, peak_mea=peak_mea)

        fit_params_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        blank_regions = 0
        for x, y in grid_walk(divs):
            if np.sum(reg_psfs[x, y]) == 0:
                blank_regions += 1
                continue
            fit_params, fit_variance = imops.fit_gauss2(reg_psfs[x, y])
            for fv in fit_variance:
                assert fv < 0.01
            fit_params_sum += np.array(fit_params)
        assert blank_regions <= 2
        fit_params_mean = fit_params_sum / ((divs * divs) - blank_regions)
        true_params["std_x"]["tgt"] = psf_std
        true_params["std_y"]["tgt"] = psf_std
        _compare_fit_params(true_params, fit_params_mean)

    def it_can_calibrate_psf_uniform_im_w_large_psf_std():
        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=400)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=tgt_mean, std=tgt_std)

        psf_std = 2.5
        peaks.widths_uniform(psf_std)
        imgs = s.render_chcy()
        bg_mean, bg_std = worker._background_estimate(imgs[0, 0], divs)
        im_sub = worker._background_subtraction(imgs[0, 0], bg_mean)

        locs, reg_psfs = worker._calibrate_psf_im(
            im_sub, divs=divs, peak_mea=peak_mea, locs=None
        )

        fit_params_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        blank_regions = 0
        for x, y in grid_walk(divs):
            if np.sum(reg_psfs[x, y]) == 0:
                blank_regions += 1
                continue
            fit_params, fit_variance = imops.fit_gauss2(reg_psfs[x, y])
            for fv in fit_variance:
                assert fv < 0.075
            fit_params_sum += np.array(fit_params)
        fit_params_mean = fit_params_sum / ((divs * divs) - blank_regions)
        true_params["std_x"]["tgt"] = psf_std
        true_params["std_y"]["tgt"] = psf_std
        _compare_fit_params(true_params, fit_params_mean)

    def it_can_calibrate_psf_im_nonuniform():
        n_z_slices = 2
        z_stack = np.array([])

        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=400)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=tgt_mean, std=tgt_std)

        templist = []
        std_most = None
        std_test_corner = None
        for z_i in range(0, n_z_slices):
            std_used = 0.5 * (2 + z_i)
            if not std_most:
                std_most = std_used
            elif not std_test_corner:
                std_test_corner = std_used
            peaks.widths_uniform(std_used)
            imgs = s.render_chcy()
            bg_mean, bg_std = worker._background_estimate(imgs[0, 0], divs)
            im_sub = worker._background_subtraction(imgs[0, 0], bg_mean)
            templist.append(im_sub)
        z_stack = np.array(templist)
        # here we make an image w nonuniform std by using data from last image
        # in z_stack to replace data of 0th image, but only in one corner
        test_corner_width = 2
        corner_xlim = test_corner_width * z_stack[0].shape[0] / divs
        corner_ylim = test_corner_width * z_stack[0].shape[1] / divs
        for x in range(0, z_stack[0].shape[0]):
            for y in range(0, z_stack[0].shape[1]):
                if (x < corner_xlim) and (y < corner_ylim):
                    z_stack[0][x][y] = z_stack[n_z_slices - 1][x][y]
        # verify that at least 1 loc is within our corner with higher std,
        # otherwise some of the later code will error out anyway and this
        # should make the problem easier to debug
        locs, reg_psfs = worker._calibrate_psf_im(
            z_stack[0], divs=divs, peak_mea=peak_mea
        )
        nbr_in_test_corner_field = 0
        for x, y in locs:
            if (x < corner_xlim) and (y < corner_ylim):
                nbr_in_test_corner_field += 1
        assert nbr_in_test_corner_field > 0
        # see if our params for "most" of the image, and the other
        # part in the corner (with higher std) are close to the true answer
        fit_params_sum_most = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fit_params_sum_test_corner = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        divisor_most = 0
        divisor_test_corner = 0
        for x, y in grid_walk(divs):
            if (
                np.sum(reg_psfs[x, y]) == 0
            ):  # cannot use imops.fit_gauss2 on all-zero psf
                continue
            fit_params, fit_variance = imops.fit_gauss2(reg_psfs[x, y])
            for fv in fit_variance:
                assert fv < 0.01
            if (x < test_corner_width) and (y < test_corner_width):
                divisor_test_corner += 1
                fit_params_sum_test_corner += np.array(fit_params)
            else:
                divisor_most += 1
                fit_params_sum_most += np.array(fit_params)
        assert ((divs * divs) - divisor_most - divisor_test_corner) <= 2
        fit_params_mean_most = fit_params_sum_most / divisor_most
        fit_params_mean_test_corner = fit_params_sum_test_corner / divisor_test_corner
        true_params["std_x"]["tgt"] = std_most
        true_params["std_y"]["tgt"] = std_most
        _compare_fit_params(true_params, fit_params_mean_most)
        true_params["std_x"]["tgt"] = std_test_corner
        true_params["std_y"]["tgt"] = std_test_corner
        _compare_fit_params(true_params, fit_params_mean_test_corner)

    def it_can_calib_psf_stats_one_channel_one_cycle():
        n_z_slices = 20
        n_fields = 1
        z_stack = np.array([])
        calib = Calibration()
        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=400)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=tgt_mean, std=tgt_std)

        templist = []
        center_z = ceil((n_z_slices - 1) / 2)
        # e.g. if n_z_slices=3, z's are [0,1,2], center_z = 1
        # idea here is to mimic having one z-slice which is most
        # focused by having its std be smaller, and then z-slices
        # further away from it have higher std
        for z_i in range(0, n_z_slices):
            std_used = 0.5 * (1 + abs(center_z - z_i))
            peaks.widths_uniform(std_used)
            imgs = s.render_chcy()
            templist.append(imgs[0][0])
        z_stack = np.array([templist])
        z_and_region_to_psf = np.array(
            worker._calib_psf_stats_one_channel(
                z_stack, n_fields, n_z_slices, calib, divs, peak_dim
            )
        )
        # TODO: find a more reasoned tgt and range for std
        std_min = 0.5
        std_max = 0.5 * (1 + center_z)
        std_range = 0.5 * (std_max - std_min)
        std_tgt = std_min + std_range
        true_params["std_x"]["tgt"] = std_tgt
        true_params["std_x"]["range"] = std_range
        true_params["std_y"]["tgt"] = std_tgt
        true_params["std_y"]["range"] = std_range
        for x, y in grid_walk(divs):
            fit_params_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            divisor = 0
            empty_psfs = 0
            for z_i in range(0, len(z_and_region_to_psf)):
                if (
                    np.sum(z_and_region_to_psf[z_i, x, y]) == 0
                ):  # cannot use imops.fit_gauss2 on all-zero psf
                    empty_psfs += 1
                    continue
                fit_params, fit_variance = imops.fit_gauss2(
                    z_and_region_to_psf[z_i, x, y]
                )
                for fv in fit_variance:
                    assert fv < 0.1
                divisor += 1
                fit_params_sum += np.array(fit_params)
            assert empty_psfs <= 2
            fit_params_mean = fit_params_sum / divisor
            _compare_fit_params(true_params, fit_params_mean)

    def it_can_calibrate_foreground_one_channel():
        calib = Calibration()
        n_z_slices = 15
        n_fields = 2
        ch_i = 0
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        ims = ims_import_result.ims[:, :, :]
        calib = worker.calibrate_background_stats(calib, ims, divs)
        calib = worker.calibrate_psf(calib, ims_import_result, divs, peak_mea)
        calib.add(
            {
                f"regional_illumination_balance.instrument_channel[{ch_i}]": np.ones(
                    (divs, divs)
                ).tolist()
            }
        )
        sigproc_params = SigprocV2Params(
            calibration_file="./bogus/calib/file/location",
            instrument_subject_id=None,
            radiometry_channels=dict(ch=ch_i),
            mode="z_stack",
        )
        fl_radmat, fl_loc = worker._calibrate_foreground_one_channel(
            calib, ims_import_result, n_fields, ch_i, sigproc_params
        )
        assert fl_radmat.shape[0] == fl_loc.shape[0] > 100
        assert fl_radmat.shape[1] == 1
        assert fl_radmat.shape[2] == n_z_slices
        assert fl_radmat.shape[3] == fl_loc.shape[1] == n_fields

        def it_can_calibrate_filter_locs_by_snr():
            sig, locs = worker._calibrate_filter_locs_by_snr(
                fl_radmat, fl_loc, n_z_slices, ch_i
            )
            assert sig.shape[0] == locs.shape[0]
            assert fl_radmat.shape[0] == fl_loc.shape[0]
            assert (
                sig.shape[0] < fl_radmat.shape[0] * n_z_slices
            )  # i.e. it filtered something

            def it_can_calibrate_balance_calc():

                balance = worker._calibrate_balance_calc(
                    ims_import_result, divs, sig, locs
                )
                assert np.min(balance) == 1  # the brightest area balanced at value 1
                assert np.count_nonzero(
                    balance == 1
                )  # i.e. only single spot is brightest

            zest()

        zest()

    def it_can_add_regional_bg_stats_to_calib():

        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=400)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            ims = s.render_flchcy()
        calib = Calibration()
        calib = worker.add_regional_bg_stats_to_calib(ims, 0, 1, divs, calib)

        check.list_t(calib["regional_bg_mean.instrument_channel[0]"], list)
        rbm_arr = np.array(calib["regional_bg_mean.instrument_channel[0]"])
        check.array_t(rbm_arr, shape=(divs, divs), dtype=np.float64)

        check.list_t(calib["regional_bg_std.instrument_channel[0]"], list)
        rbs_arr = np.array(calib["regional_bg_std.instrument_channel[0]"])
        check.array_t(rbs_arr, shape=(divs, divs), dtype=np.float64)

    def it_can_calibrate_background_stats():
        # other tests for the lower level functions called by
        # calibrate_background_states are checking for the ability
        # to measure background noise accurately, handle nonuniformity
        # in the image, etc.  This test is just checking that the
        # data returned is getting put into the calibration objects
        # with the right shape, and using the right keyword
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=400)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            ims = s.render_flchcy()
        calib = Calibration()
        calib = worker.calibrate_background_stats(calib, ims, divs)

        check.list_t(calib["regional_bg_mean.instrument_channel[0]"], list)
        rbm_arr = np.array(calib["regional_bg_mean.instrument_channel[0]"])
        check.array_t(rbm_arr, shape=(divs, divs), dtype=np.float64)

        check.list_t(calib["regional_bg_std.instrument_channel[0]"], list)
        rbs_arr = np.array(calib["regional_bg_std.instrument_channel[0]"])
        check.array_t(rbs_arr, shape=(divs, divs), dtype=np.float64)

    def it_can_calibrate_psfs():
        # other tests for the lower level functions called by calibrate_psf()
        # are checking for the ability to return accurate measures of std,
        # handle nonuniform images, etc.  This test is just checking that
        # the data received is getting put into the calibration object in
        # the correct shape and with the correct keyword

        calib = Calibration()
        n_z_slices = 20
        n_fields = 3
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        calib = worker.calibrate_psf(calib, ims_import_result, divs, peak_mea)

        check.list_t(calib["regional_psf_zstack.instrument_channel[0]"], list)
        rbm_arr = np.array(calib["regional_psf_zstack.instrument_channel[0]"])
        check.array_t(
            rbm_arr, shape=(None, divs, divs, peak_mea, peak_mea), dtype=np.float64
        )

    def it_can_calibrate_regional_illumination_balance():
        calib = Calibration()
        n_z_slices = 15
        n_fields = 2
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        ims = ims_import_result.ims[:, :, :]
        calib = worker.calibrate_background_stats(calib, ims, divs)
        calib = worker.calibrate_psf(calib, ims_import_result, divs, peak_mea)
        calib = worker.calibrate_regional_illumination_balance(
            calib, ims_import_result, divs, peak_mea
        )
        check.list_t(calib["regional_illumination_balance.instrument_channel[0]"], list)
        rib_arr = np.array(calib["regional_illumination_balance.instrument_channel[0]"])
        check.array_t(rib_arr, shape=(divs, divs), dtype=np.float64)

    zest()


# @zest.skip(reason="Not ready")
# @zest.group("integration")
# def zest_sigproc_v2_calibration():
#     """
#     This is an integration test of the entire sigproc_v2 pipeline
#     with synthetic data from calibration to the calls.
#
#     Some of these tests use unrealisic conditions (called "syncon")
#     such as perfect isolation of peaks so that there is no stochasitc behavior;
#     other tests allow stochastic behavior and check bounds of behavior which
#     is less reliable.
#     """
#
#     def it_calibrates_syncon_grid():
#         s = synth.Synth(overwrite=True)
#         peaks = (
#             synth.PeaksModelPSF(n_peaks=2300, depth_in_microns=0.3)
#             .locs_grid(steps=50)
#             .amps_randomize(mean=1000, std=0)
#             .remove_near_edges()
#         )
#         synth.CameraModel(bias=100, std=10)
#
#         flchcy_ims = s.render_flchcy()
#         calib = Calibration()
#
#         divs = 5
#         worker._calibrate(flchcy_ims, calib, divs=divs)
#
#         assert np.array(calib["regional_bg_mean.instrument_channel[0]"]).shape == (
#             divs,
#             divs,
#         )
#         assert np.array(calib["regional_bg_std.instrument_channel[0]"]).shape == (
#             divs,
#             divs,
#         )
#         assert np.array(
#             calib["regional_illumination_balance.instrument_channel[0]"]
#         ).shape == (divs, divs)
#         assert np.array(calib["regional_psf_zstack.instrument_channel[0]"]).shape == (
#             1,
#             divs,
#             divs,
#             11,
#             11,
#         )
#
#         # Using that calibration on a new dataset, see if it recovers the
#         # amplitudes well
#         s = synth.Synth(overwrite=True)
#         peaks = (
#             synth.PeaksModelPSF(n_peaks=1000, depth_in_microns=0.3)
#             .locs_randomize()
#             .amps_randomize(mean=1000, std=0)
#             .remove_near_edges()
#         )
#         synth.CameraModel(bias=100, std=10)
#         chcy_ims = s.render_chcy()
#
#         sigproc_params = SigprocV2Params(
#             calibration=calib,
#             instrument_subject_id=None,
#             radiometry_channels=dict(ch_0=0),
#         )
#         chcy_ims, locs, radmat, aln_offsets, aln_scores = worker.sigproc_field(
#             chcy_ims, sigproc_params
#         )
#
#         # TODO: assert centered around 1000
#
#     # def it_compensates_for_regional_psf_differences():
#     #     raise NotImplementedError
#     #
#     # def alarms():
#     #     def it_alarms_if_background_significantly_different_than_calibration():
#     #         raise NotImplementedError
#     #
#     #     def it_alarms_if_psf_significantly_different_than_calibration():
#     #         raise NotImplementedError
#     #
#     #     zest()
#
#     zest()
""" regional_bg_mean.instrument_channel[{ch}]
regional_illumination_balance.instrument_channel[{ch}]
  like bg_mean but fg, somewhat correlated to bg
regional_psf_zstack.instrument_channel[{sigproc_params.output_channel_to_input_channel(out_ch_i)}
  when z position is optimum (ie in focus), psf looks like 2d gaussian
  in center of image, will even be a circular 2d gaussian
  even when perfectly focused, sometimes in corners will be rotated ellipse rather than circular
  given that z is not always right at focal point, may not even be gaussian
    ...because there are some interference patterns when you are out of focal planes, so you get dark/light bands
  we're going to have photographs at several z's, one of which is in optimal z (we hope)
  in calibration data, what looks like chemical cycles are actually different z slices

regional_bg_std.instrument_channel[{in_ch_i}]
  std dev of bg within a single region
"""
