from math import floor, ceil

import numpy as np
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2 import sigproc_v2_common as common
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


'''
class NulCalibSigprocV2Params(SigprocV2Params):
    """
    We need to break the chicked-and-egg problem.
    During Foreground Calibration we need to call the _sigproc_field
    function which normally uses the "regional_illumination_balance"
    calib property to correct for regional differences.
    But we need to pass in a "fake" calib "regional_illumination_balance"
    with unity values to extract the differences.
    """
    def __init__(self, ch_i):
        super().__init__()
        params = SigprocV2Params(calibration_file=None, mode="analyze",)
        params.calibration = Calibration
'''


class MockImsImportResult:
    @property
    def ims(self):
        return self.ims_to_return

    def n_fields_channel_cycles(self):
        return self.n_fields, self.n_channels, self.n_cycles

    def __init__(self, ims_to_return, n_fields, n_channels, n_cycles):
        self.ims_to_return = ims_to_return
        self.dim = ims_to_return.shape[-1]
        self.n_fields = n_fields
        self.n_channels = n_channels
        self.n_cycles = n_cycles
        self.params = ImsImportParams(is_movie=True,)

    def n_fields_channel_frames(self):
        return self.n_fields, self.n_channels, self.n_cycles


def result_from_z_stack(n_fields=1, n_channels=1, n_cycles=1, uniformity="uniform"):
    """
    Generate a MockImportResult

    At moment, always generates a regionally-uniform distribution of peaks
    over all cycles.

    TODO: Probably need to change this so that is can
        1. Generate wide->narrow->wide z_stack PSFs
        2. Regional differences.
    """

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


# @zest.skip(reason="SLOW")
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

        true_params = Munch(
            amp=Munch(tgt=1, range=0.1),
            std_x=Munch(tgt=None, range=0.15),
            std_y=Munch(tgt=None, range=0.15),
            pos_x=Munch(tgt=midpt, range=0.10),
            pos_y=Munch(tgt=midpt, range=0.10),
            rho=Munch(tgt=0, range=0.07),
            const=Munch(tgt=0, range=0.05),
            mea=Munch(tgt=peak_mea, range=0.1),
        )

    def background_stats():
        true_bg_mean = 200
        true_bg_std = 15

        def _synth(_true_bg_std, howmanydims=1):
            with synth.Synth(overwrite=True) as s:
                (
                    synth.PeaksModelGaussianCircular(n_peaks=100)
                    .locs_randomize()
                    .amps_constant(val=10000)
                )
                synth.CameraModel(bias=true_bg_mean, std=_true_bg_std)
                if howmanydims == 3:
                    ims = s.render_flchcy()
                elif howmanydims == 2:
                    ims = s.render_chcy()
                else:
                    ims = s.render_chcy()[0, 0]
            return ims

        def common_bg_stats():
            im = _synth(true_bg_std, howmanydims=1)

            est_bg_mean, est_bg_std = worker._background_estimate_im(im, divs)

            def it_estimates_uniform_background_correctly():
                # Bounds here are empirical
                assert np_within(np.mean(est_bg_mean), true_bg_mean, 1)
                assert np_within(np.mean(est_bg_std), true_bg_std, 1)

            def it_subtracts_uniform_bg_mean_correctly():
                # Bounds here are also empirical -- the 1/true_bg_std
                # is because when we background subtract the mean should be
                # close to zero.  Could use a better way to set bounds
                im_sub = worker._background_subtraction(im, est_bg_mean)
                new_est_bg_mean, new_est_bg_std = worker._background_estimate_im(
                    im_sub, divs
                )
                assert np_within(np.mean(new_est_bg_mean), 0, (1 / true_bg_std))

            def it_estimates_nonuniform_bg_mean_correctly():
                # MAKE the outside border half as light, width of one div
                im = _synth(true_bg_std, howmanydims=1)
                border = int(im.shape[0] / divs)

                ys, xs = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
                dist_to_edge = np.minimum(
                    np.minimum(xs, im.shape[1] - xs - 1),
                    np.minimum(ys, im.shape[0] - ys - 1),
                )
                scale_factor = np.ones_like(im)
                scale_factor[dist_to_edge < border] = 0.5
                im_sc = im * scale_factor

                # ESTIMATE mean and std, check that it got nonuniformity of means
                est_bg_mean, est_bg_std = worker._background_estimate_im(im_sc, divs)
                for y, x in grid_walk(divs):
                    if x in [0, divs - 1] or y in [0, divs - 1]:  # i.e. near the edge
                        assert np_within(
                            (true_bg_mean * 0.5), est_bg_mean[y, x], true_bg_std
                        )
                        assert np_within(
                            (true_bg_std * 0.5), est_bg_std[y, x], true_bg_std / 3
                        )
                    else:
                        assert np_within(true_bg_mean, est_bg_mean[y, x], true_bg_std)
                        try:
                            assert np_within(
                                true_bg_std, est_bg_std[y, x], true_bg_std / 3
                            )
                        except AssertionError:
                            debug(true_bg_std, est_bg_std[y, x], true_bg_std / 3)
                            raise

            def it_estimates_nonuniform_bg_std_correctly():
                im = _synth(true_bg_std, howmanydims=1)
                # CREATE second image with larger std, to use for border region
                true_bg_std2 = 2 * true_bg_std
                im2 = _synth(true_bg_std2, howmanydims=1)
                # MERGE the two images into one image with a nonuniform std
                border = int(im.shape[0] / divs)
                ys, xs = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
                dist_to_edge = np.minimum(
                    np.minimum(xs, im.shape[0] - xs), np.minimum(ys, im.shape[1] - ys)
                )
                im3 = np.where(dist_to_edge < border, im2, im)
                est_bg_mean, est_bg_std = worker._background_estimate_im(im3, divs)
                # ESTIMATE mean and std, check that it got nonuniformity of means
                for y, x in grid_walk(divs):
                    if x in [0, divs - 1] or y in [0, divs - 1]:  # i.e. near the edge
                        assert np_within(
                            est_bg_std[x][y], true_bg_std2, true_bg_std2 / 2
                        )
                        assert np_within(est_bg_mean[x][y], true_bg_mean, true_bg_std2)
                    else:
                        assert np_within(est_bg_std[x][y], true_bg_std, true_bg_std / 2)
                        assert np_within(est_bg_mean[x][y], true_bg_mean, true_bg_std)

            def it_subtracts_nonuniform_bg_mean_correctly():
                # MAKE the outside border half as light, width of one div
                im = _synth(true_bg_std, (0, 0))
                border = int(im.shape[0] / divs)
                ys, xs = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
                dist_to_edge = np.minimum(
                    np.minimum(xs, im.shape[1] - xs - 1),
                    np.minimum(ys, im.shape[0] - ys - 1),
                )
                scale_factor = np.ones_like(im)
                scale_factor[dist_to_edge < border] = 1 - (1 / border)
                im_sc = im * scale_factor
                # ESTIMATE mean and std, check that it gets new bg mean close to 0
                est_bg_mean, est_bg_std = worker._background_estimate_im(im_sc, divs)
                im_sub = worker._background_subtraction(im_sc, est_bg_mean)
                new_mean, new_std = worker._background_estimate_im(im_sub, divs)
                try:
                    assert np_within(np.mean(new_mean), 0, true_bg_std)
                except AssertionError:
                    debug(np.mean(new_mean), 0, true_bg_std)
                    raise

            def it_estimates_regional_bg_stats_correctly():
                ims = _synth(true_bg_std, howmanydims=2)

                est_bg_mean, est_bg_std = worker._background_stats_ims(ims, divs)
                assert len(est_bg_mean.shape) == 2
                allow_err = true_bg_mean / 5
                assert np_within(true_bg_mean, est_bg_mean.max(), allow_err)
                assert np_within(true_bg_mean, est_bg_mean.min(), allow_err)

                assert len(est_bg_std.shape) == 2
                allow_err = true_bg_std / 3
                assert np_within(true_bg_std, est_bg_std.max(), allow_err)
                assert np_within(true_bg_std, est_bg_std.min(), allow_err)

            def it_handles_all_zeros():
                im = np.zeros((1, 1, 512, 512))
                est_bg_mean, est_bg_std = worker._background_stats_ims(im, divs)
                assert np.all(est_bg_mean == 0)
                assert np.all(est_bg_std == 0)

            def it_handles_all_noise():
                with synth.Synth(overwrite=True) as s:
                    synth.CameraModel(bias=true_bg_mean, std=true_bg_std)
                    im = s.render_chcy()
                est_bg_mean, est_bg_std = worker._background_stats_ims(im, divs)
                allow_err = true_bg_mean / 10
                assert np_within(true_bg_mean, est_bg_mean.max(), allow_err)
                assert np_within(true_bg_mean, est_bg_mean.min(), allow_err)
                allow_err = true_bg_std / 10
                assert np_within(true_bg_std, est_bg_std.max(), allow_err)
                assert np_within(true_bg_std, est_bg_std.min(), allow_err)

            def it_can_calibrate_background_stats():
                # other tests for the lower level functions called by
                # calibrate_background_states are checking for the ability
                # to measure background noise accurately, handle nonuniformity
                # in the image, etc.  This test is just checking that the
                # data returned is getting put into the calibration objects
                # with the right shape, and using the right keyword
                ims_import_result = result_from_z_stack(n_fields=2, n_cycles=15)
                calib = Calibration()
                sigproc_v2_params = SigprocV2Params(
                    calibration_file="temp", mode=common.SIGPROC_V2_INSTRUMENT_CALIB
                )
                sigproc_v2_params.mode = common.SIGPROC_V2_INSTRUMENT_ANALYZE
                sigproc_v2_params.calibration = calib
                calib = worker._calibrate_step_1_background_stats(
                    calib, ims_import_result, sigproc_v2_params
                )

                check.list_t(calib["regional_bg_mean.instrument_channel[0]"], list)
                rbm_arr = np.array(calib["regional_bg_mean.instrument_channel[0]"])
                check.array_t(rbm_arr, shape=(divs, divs), dtype=np.float64)

                check.list_t(calib["regional_bg_std.instrument_channel[0]"], list)
                rbs_arr = np.array(calib["regional_bg_std.instrument_channel[0]"])
                check.array_t(rbs_arr, shape=(divs, divs), dtype=np.float64)

            zest()

        zest()

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

    def psf_stats():
        true_bg_mean = 200
        true_bg_std = 15

        def _synth_psf(_true_bg_std, _true_psf_std, howmanydims=1):
            with synth.Synth(overwrite=True) as s:
                peaks = (
                    synth.PeaksModelGaussianCircular(n_peaks=400)
                    .locs_randomize()
                    .amps_constant(val=10000)
                )
                synth.CameraModel(bias=true_bg_mean, std=_true_bg_std)
                peaks.widths_uniform(_true_psf_std)
                if howmanydims == 3:
                    ims = s.render_flchcy()
                elif howmanydims == 2:
                    ims = s.render_chcy()
                else:
                    ims = s.render_chcy()[0, 0]
            return ims

        def it_can_calibrate_psf_uniform_im():
            # CREATE image with known PSF std
            true_psf_std = 0.5
            img = _synth_psf(true_bg_std, true_psf_std, howmanydims=1)
            # ESTIMATE locs and psfs for this image
            est_bg_mean, est_bg_std = worker._background_estimate_im(img, divs)
            im_sub = worker._background_subtraction(img, est_bg_mean)
            locs, reg_psfs = worker._psf_extract(im_sub, divs=divs, peak_mea=peak_mea)
            # CALCULATE psfs using fit_gauss2()
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
            # COMPARE fitted parameters to targets
            fit_params_mean = fit_params_sum / ((divs * divs) - blank_regions)
            true_params["std_x"]["tgt"] = true_psf_std
            true_params["std_y"]["tgt"] = true_psf_std
            _compare_fit_params(true_params, fit_params_mean)

        def it_can_calibrate_psf_uniform_im_w_large_psf_std():
            # CREATE image with known PSF std, much larger than in previous test
            true_psf_std = 2.5
            img = _synth_psf(true_bg_std, true_psf_std, howmanydims=1)
            # ESTIMATE locs and psfs for this image
            est_bg_mean, est_bg_std = worker._background_estimate_im(img, divs)
            im_sub = worker._background_subtraction(img, est_bg_mean)
            locs, reg_psfs = worker._psf_extract(im_sub, divs=divs, peak_mea=peak_mea)
            # CALCULATE psfs using fit_gauss2()
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
            # COMPARE fitted parameters to targets
            fit_params_mean = fit_params_sum / ((divs * divs) - blank_regions)
            true_params["std_x"]["tgt"] = true_psf_std
            true_params["std_y"]["tgt"] = true_psf_std
            _compare_fit_params(true_params, fit_params_mean)

        def it_can_calibrate_psf_im_nonuniform():
            # CREATE image with known small PSF std
            true_sm_psf_std = 0.5
            img1 = _synth_psf(true_bg_std, true_sm_psf_std, howmanydims=1)
            est_bg_mean, est_bg_std = worker._background_estimate_im(img1, divs)
            im_most = worker._background_subtraction(img1, est_bg_mean)
            # CREATE second image with known large PSF std
            true_lg_psf_std = 2.5
            img2 = _synth_psf(true_bg_std, true_lg_psf_std, howmanydims=1)
            est_bg_mean, est_bg_std = worker._background_estimate_im(img2, divs)
            im_corner = worker._background_subtraction(img2, est_bg_mean)
            # CREATE hybrid with large PSF std in one corner
            corner_divs = 2
            corner_size = corner_divs * int(im_most.shape[0] / divs)
            ys, xs = np.meshgrid(
                np.arange(im_most.shape[0]), np.arange(im_most.shape[1])
            )
            dist_to_corner = np.maximum(xs, ys)
            im3 = np.where(dist_to_corner < corner_size, im_corner, im_most)
            # VERIFY that at least 1 loc is within our corner with higher std,
            # otherwise some of the later code will error out anyway and this
            # should make the problem easier to debug
            locs, reg_psfs = worker._psf_extract(im3, divs=divs, peak_mea=peak_mea)
            nbr_in_test_corner_field = 0
            for x, y in locs:
                if (x < corner_size) and (y < corner_size):
                    nbr_in_test_corner_field += 1
            assert nbr_in_test_corner_field > 0
            # VERIFY that our params for "most" of the image, and the other
            # part in the corner (with higher std) are close to the true answer
            fit_params_sum_most = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            fit_params_sum_test_corner = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
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
                if (x < corner_divs) and (y < corner_divs):
                    divisor_test_corner += 1
                    fit_params_sum_test_corner += np.array(fit_params)
                else:
                    divisor_most += 1
                    fit_params_sum_most += np.array(fit_params)
            assert ((divs * divs) - divisor_most - divisor_test_corner) <= 2
            fit_params_mean_most = fit_params_sum_most / divisor_most
            fit_params_mean_test_corner = (
                fit_params_sum_test_corner / divisor_test_corner
            )
            true_params["std_x"]["tgt"] = true_sm_psf_std
            true_params["std_y"]["tgt"] = true_sm_psf_std
            true_params["std_x"]["range"] = 1
            true_params["std_y"]["range"] = 1
            _compare_fit_params(true_params, fit_params_mean_most)
            true_params["std_x"]["tgt"] = true_lg_psf_std
            true_params["std_y"]["tgt"] = true_lg_psf_std
            _compare_fit_params(true_params, fit_params_mean_test_corner)

        zest()

    @zest.skip(reason="Need to ask Ross about underdefined errors I think there was a merge collision")
    def it_can_calib_psf_stats_one_channel_one_cycle():
        # CREATE a set of peaks which will appear in all images
        n_z_slices = 20
        n_fields = 1
        z_stack = np.array([])
        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=400)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=bg_mean, std=bg_std)
        # GENERATE a z-stack of images with the same peaks, different psf stds
        templist = []
        center_z = ceil((n_z_slices - 1) / 2)
        # e.g. if n_z_slices=3, z's are [0,1,2], center_z = 1
        # idea here is to mimic having one z-slice which is most
        # focused by having its std be smaller, and then z-slices
        # further away from it have higher std
        for z_i in range(0, n_z_slices):
            std_used = std_min * (1 + abs(center_z - z_i))
            std_max = max(std_used, std_max)
            peaks.widths_uniform(std_used)
            imgs = s.render_chcy()
            templist.append(imgs[0][0])
        z_stack = np.array([templist])

        # MOCK a SigprocV2Param
        sigproc_v2_params = SigprocV2Params(
            calibration_file="temp", mode=common.SIGPROC_V2_INSTRUMENT_CALIB
        )
        sigproc_v2_params.mode = common.SIGPROC_V2_INSTRUMENT_ANALYZE
        calib = Calibration()
        sigproc_v2_params.calibration = calib

        # MEASURE the psf stats using this z-stack
        z_and_region_to_psf = np.array(
            worker._psf_stats_one_channel(z_stack, sigproc_v2_params)
        )
        # DETERMINE if we measured things successfully
        fit_params_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        divisor = 0
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
                fit_params, _ = imops.fit_gauss2(z_and_region_to_psf[z_i, x, y])
                divisor += 1
                fit_params_sum += np.array(fit_params)

        assert empty_psfs <= 2
        fit_params_mean = fit_params_sum / divisor
        # TODO: find a more reasoned range for std
        std_range = 0.5 * (std_max - std_min)
        true_params["std_x"]["tgt"] = std_min
        true_params["std_x"]["range"] = std_range
        true_params["std_y"]["tgt"] = std_min
        true_params["std_y"]["range"] = std_range
        _compare_fit_params(true_params, fit_params_mean)

    def it_can_calibrate_foreground_one_channel():
        # MOCK a z_stack ImsImportResult
        n_z_slices = 15
        n_fields = 2
        ch_i = 0
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        # MOCK a SigprocV2Param and calibration
        sigproc_v2_params = SigprocV2Params(
            calibration_file="temp", mode=common.SIGPROC_V2_INSTRUMENT_CALIB
        )
        sigproc_v2_params.mode = common.SIGPROC_V2_INSTRUMENT_ANALYZE
        calib = Calibration()
        sigproc_v2_params.calibration = calib

        # PROCESS the calib object through steps 1 and 2
        calib = worker._calibrate_step_1_background_stats(
            calib, ims_import_result, sigproc_v2_params
        )
        calib = worker._calibrate_step_2_psf(
            calib, ims_import_result, sigproc_v2_params
        )
        calib.add(
            {
                f"regional_illumination_balance.instrument_channel[{ch_i}]": np.ones(
                    (divs, divs)
                ).tolist()
            }
        )

        # FIND radmat and locs
        fl_radmat, fl_loc = worker._foreground_stats(
            calib, ims_import_result, n_fields, ch_i, sigproc_v2_params
        )
        try:
            assert (
                fl_radmat.shape[0] == fl_loc.shape[0] > 100
            )  # found at least 100 locs
        except AssertionError:
            debug("did not find enough locs", fl_loc.shape[0])
            raise
        assert fl_radmat.shape[1] == 1
        assert fl_radmat.shape[2] == n_z_slices
        assert fl_radmat.shape[3] == fl_loc.shape[1] == n_fields

        def it_can_calibrate_filter_locs_by_snr():
            snr_min = 2
            sig_min = 10
            sig_max = 100000
            # REUSE radmat and locs from previous test
            sig, locs = worker._foreground_filter_locs(
                fl_radmat, fl_loc, n_z_slices, ch_i, snr_min, sig_min, sig_max
            )
            assert sig.shape[0] == locs.shape[0]
            assert fl_radmat.shape[0] == fl_loc.shape[0]
            assert (
                sig.shape[0] < fl_radmat.shape[0] * n_z_slices
            )  # i.e. it filtered something

            def it_can_calibrate_balance_calc():
                # REUSE ImsImportResult and filtered sig, locs from previous test
                balance = worker._foreground_balance(ims_import_result, divs, sig, locs)
                assert np.min(balance) == 1  # the brightest area balanced at value 1
                assert np.count_nonzero(
                    balance == 1
                )  # i.e. only single spot is brightest

            zest()

        def it_raises_on_low_counts():
            with zest.raises(ValueError, in_args="filter retained less than"):
                worker._foreground_filter_locs(
                    fl_radmat,
                    fl_loc,
                    n_z_slices,
                    ch_i,
                    snr_min=100,
                    sig_min=1e6,
                    sig_max=0,
                )

        zest()

    def it_can_calibrate_psfs():
        # other tests for the lower level functions called by calibrate_psf()
        # are checking for the ability to return accurate measures of std,
        # handle nonuniform images, etc.  This test is just checking that
        # the data received is getting put into the calibration object in
        # the correct shape and with the correct keyword
        # CREATE ims_import_result object
        n_z_slices = 20
        n_fields = 3
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        # MOCK a SigprocV2Param and calib
        sigproc_v2_params = SigprocV2Params(
            calibration_file="temp", mode=common.SIGPROC_V2_INSTRUMENT_CALIB
        )
        sigproc_v2_params.mode = common.SIGPROC_V2_INSTRUMENT_ANALYZE
        calib = Calibration()
        sigproc_v2_params.calibration = calib
        calib = worker._calibrate_step_2_psf(
            calib, ims_import_result, sigproc_v2_params
        )
        # VERIFY that the calib object got the right kinds of things added to it
        check.list_t(calib["regional_psf_zstack.instrument_channel[0]"], list)
        rbm_arr = np.array(calib["regional_psf_zstack.instrument_channel[0]"])
        check.array_t(
            rbm_arr, shape=(None, divs, divs, peak_mea, peak_mea), dtype=np.float64
        )

    def it_can_calibrate_regional_illumination_balance():
        # CREATE calib and ImsImportResult
        calib = Calibration()
        n_z_slices = 15
        n_fields = 2
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        ims = ims_import_result.ims[:, :, :]
        # MOCK a SigprocV2Param
        sigproc_v2_params = SigprocV2Params(
            calibration_file="temp", mode=common.SIGPROC_V2_INSTRUMENT_CALIB
        )
        sigproc_v2_params.mode = common.SIGPROC_V2_INSTRUMENT_ANALYZE
        sigproc_v2_params.calibration = calib

        calib = worker._calibrate_step_1_background_stats(
            calib, ims_import_result, sigproc_v2_params
        )
        # calib = worker._calibrate_background_stats(calib, ims, divs)
        calib = worker._calibrate_step_2_psf(
            calib, ims_import_result, sigproc_v2_params
        )
        # calib = worker._calibrate_psf(calib, ims_import_result.ims, divs, peak_mea)
        nbr_failures = 0
        nbr_successes = 0

        while True:
            try:
                # calib = worker._calibrate_regional_illumination_balance(
                #    calib, ims_import_result, divs, peak_mea
                # )
                calib = worker._calibrate_step_3_regional_illumination_balance(
                    calib, ims_import_result, sigproc_v2_params
                )
            except AssertionError:
                nbr_failures += 1
            else:
                nbr_successes += 1
            if nbr_failures > 1 or nbr_successes > 5 * nbr_failures:
                break
        assert nbr_failures < 2

        # VERIFY that _calibrate_regional_illumination_balance() puts data of the right
        # type and shape into the right place in the calib object.  Other tests will
        # test the lower level functions.
        check.list_t(calib["regional_illumination_balance.instrument_channel[0]"], list)
        rib_arr = np.array(calib["regional_illumination_balance.instrument_channel[0]"])
        check.array_t(rib_arr, shape=(divs, divs), dtype=np.float64)

    zest()
