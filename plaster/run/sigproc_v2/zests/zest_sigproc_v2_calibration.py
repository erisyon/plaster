from munch import Munch
import numpy as np
from math import ceil
from plumbum import local

from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.image import imops
from plaster.tools.utils.utils import np_within
from plaster.tools.schema import check
from plaster.run.base_result import BaseResult, ArrayResult
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.tools.utils.fancy_indexer import FancyIndexer
from plaster.tools.log.log import debug

from zest import zest


class MockImsImportResult:
    @property
    def ims(self):
        return self.ims_to_return

    def n_fields_channel_cycles(self):
        return self.n_fields, self.n_channels, self.n_cycles

    def __init__(self, ims_to_return, n_fields, n_channels, n_cycles, is_movie=True):
        self.ims_to_return = ims_to_return
        self.n_fields = n_fields
        self.n_channels = n_channels
        self.n_cycles = n_cycles
        self.params = ImsImportParams(is_movie=is_movie)


def result_from_z_stack(n_fields=1, n_channels=1, n_cycles=1, uniformity="uniform"):
    """
    Generate a MockImportResult

    At moment, always generates a regionally-uniform distribution of peaks
    over all cycles.

    TODO: Probably need to change this so that is can
        1. Generate wide->narrow->wide z_stack PSFs
        2. Regionaly differences.
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

    # STUFF imgs into a MockImsImportResult
    mock_ims_import_result = MockImsImportResult(
        ims_to_return=imgs, n_fields=n_fields, n_channels=n_channels, n_cycles=n_cycles
    )
    return mock_ims_import_result


def grid_walk(divs):
    """
    Helper to traverse 2 equal dimensions
    """
    for y in range(0, divs):
        for x in range(0, divs):
            yield y, x


def zest_sigproc_v2_calibration():
    divs = None
    bg_mean = None
    bg_std = None
    peak_mea = None
    peak_dim = None
    midpt = None
    true_params = {}  # if None pylint gets confused

    def _before():
        nonlocal divs, bg_mean, bg_std, peak_mea, midpt, true_params, peak_dim
        divs = 5
        bg_mean = 100
        bg_std = 10
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
                if howmanydims==3:
                    ims = s.render_flchcy()
                elif howmanydims == 2:
                    ims = s.render_chcy()
                else:
                    ims = s.render_chcy()[0,0]
            return ims

        def common_bg_stats():
            im = _synth(true_bg_std,howmanydims=1)

            est_bg_mean, est_bg_std = worker._background_estimate(im, divs)

            def it_estimates_uniform_background_correctly():
                # Bounds here are empirical
                assert np_within(np.mean(est_bg_mean), true_bg_mean, 1)
                assert np_within(np.mean(est_bg_std), true_bg_std, 1)

            def it_subtracts_uniform_bg_mean_correctly():
                # Bounds here are also empirical -- the 1/true_bg_std
                # is because when we background subtract the mean should be
                # close to zero.  Could use a better way to set bounds
                im_sub = worker._background_subtraction(im, est_bg_mean)
                new_est_bg_mean, new_est_bg_std = worker._background_estimate(
                    im_sub, divs
                )
                assert np_within(np.mean(new_est_bg_mean), 0, (1 / true_bg_std))

            def it_estimates_nonuniform_bg_mean_correctly():
                # MAKE the outside border half as light, width of one div
                im = _synth(true_bg_std,howmanydims=1)
                border = int(im.shape[0] / divs)

                ys, xs = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
                dist_to_edge = np.minimum(np.minimum(xs, im.shape[1] - xs - 1), np.minimum(ys, im.shape[0] - ys - 1))
                scale_factor = np.ones_like(im)
                scale_factor[dist_to_edge < border] = 0.5
                im_sc = im * scale_factor

                # ESTIMATE mean and std, check that it got nonuniformity of means
                est_bg_mean, est_bg_std = worker._background_estimate(im_sc, divs)
                for y, x in grid_walk(divs):
                    if x in [0, divs - 1] or y in [0, divs - 1]: #i.e. near the edge
                        assert np_within((true_bg_mean * 0.5), est_bg_mean[y, x], true_bg_std)
                        assert np_within((true_bg_std * 0.5), est_bg_std[y, x], true_bg_std / 3)
                    else:
                        assert np_within(true_bg_mean, est_bg_mean[y, x], true_bg_std)
                        try:
                            assert np_within(true_bg_std, est_bg_std[y, x], true_bg_std / 3)
                        except AssertionError:
                            debug(true_bg_std, est_bg_std[y, x], true_bg_std / 3)
                            raise

            def it_estimates_nonuniform_bg_std_correctly():
                im = _synth(true_bg_std,howmanydims=1)
                # CREATE second image with larger std, to use for border region
                true_bg_std2 = 2*true_bg_std
                im2 = _synth(true_bg_std2,howmanydims=1)
                # MERGE the two images into one image with a nonuniform std
                border = int(im.shape[0] / divs)
                ys, xs = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
                dist_to_edge = np.minimum(np.minimum(xs, im.shape[0] - xs), np.minimum(ys, im.shape[1] - ys))
                im3 = np.where(dist_to_edge < border,im2,im)
                est_bg_mean, est_bg_std = worker._background_estimate(im3, divs)
                # ESTIMATE mean and std, check that it got nonuniformity of means
                for y, x in grid_walk(divs):
                    if x in [0, divs - 1] or y in [0, divs - 1]: #i.e. near the edge
                        assert np_within(est_bg_std[x][y], true_bg_std2, true_bg_std2 / 2)
                        assert np_within(est_bg_mean[x][y], true_bg_mean, true_bg_std2)
                    else:
                        assert np_within(est_bg_std[x][y], true_bg_std, true_bg_std / 2)
                        assert np_within(est_bg_mean[x][y], true_bg_mean, true_bg_std)

            def it_subtracts_nonuniform_bg_mean_correctly():
                # MAKE the outside border half as light, width of one div
                im = _synth(true_bg_std,(0,0))
                border = int(im.shape[0] / divs)
                ys, xs = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
                dist_to_edge = np.minimum(np.minimum(xs, im.shape[1] - xs - 1), np.minimum(ys, im.shape[0] - ys - 1))
                scale_factor = np.ones_like(im)
                scale_factor[dist_to_edge < border] = 1 - (1/border)
                im_sc = im * scale_factor
                # ESTIMATE mean and std, check that it gets new bg mean close to 0
                est_bg_mean, est_bg_std = worker._background_estimate(im_sc, divs)
                im_sub = worker._background_subtraction(im_sc, est_bg_mean)
                new_mean, new_std = worker._background_estimate(im_sub, divs)
                try:
                    assert np_within(np.mean(new_mean), 0, true_bg_std)
                except AssertionError:
                    debug(np.mean(new_mean), 0, true_bg_std)
                    raise

            def it_adds_regional_bg_stats_to_calib_correctly():
                ims = _synth(true_bg_std,howmanydims=3)
                calib = Calibration()
                calib = worker.add_regional_bg_stats_to_calib(ims, 0, 1, divs, calib)

                est_bg_mean = np.array(calib["regional_bg_mean.instrument_channel[0]"])
                assert len(est_bg_mean.shape) == 2
                allow_err = true_bg_mean/5
                assert np_within(true_bg_mean,est_bg_mean.max(),allow_err)
                assert np_within(true_bg_mean,est_bg_mean.min(),allow_err)

                est_bg_std = np.array(calib["regional_bg_std.instrument_channel[0]"])
                assert len(est_bg_std.shape) == 2
                allow_err = true_bg_std/3
                assert np_within(true_bg_std,est_bg_std.max(),allow_err)
                assert np_within(true_bg_std,est_bg_std.min(),allow_err)

            def it_can_calibrate_background_stats():
                # other tests for the lower level functions called by
                # calibrate_background_states are checking for the ability
                # to measure background noise accurately, handle nonuniformity
                # in the image, etc.  This test is just checking that the
                # data returned is getting put into the calibration objects
                # with the right shape, and using the right keyword
                ims = _synth(true_bg_std,howmanydims=3)

                calib = Calibration()
                calib = worker.calibrate_background_stats(calib, ims, divs)

                check.list_t(calib["regional_bg_mean.instrument_channel[0]"], list)
                rbm_arr = np.array(calib["regional_bg_mean.instrument_channel[0]"])
                check.array_t(rbm_arr, shape=(divs, divs), dtype=np.float64)

                check.list_t(calib["regional_bg_std.instrument_channel[0]"], list)
                rbs_arr = np.array(calib["regional_bg_std.instrument_channel[0]"])
                check.array_t(rbs_arr, shape=(divs, divs), dtype=np.float64)

            zest()

        zest()

    def _compare_fit_params(true_params, fit_params):
        for ix, parm in enumerate(true_params.keys()):
            try:
                assert true_params[parm]["range"] > abs(
                    true_params[parm]["tgt"] - fit_params[ix]
                )
            except AssertionError:
                tgt = true_params[parm]["tgt"]
                actual = fit_params[ix]
                range_ = true_params[parm]["range"]
                debug(parm, tgt, actual, range_)
                raise

    def psf_stats():
        true_bg_mean = 200
        true_bg_std = 15

        def _synth_psf(_true_bg_std,_true_psf_std,howmanydims=1):
            with synth.Synth(overwrite=True) as s:
                peaks = (
                    synth.PeaksModelGaussianCircular(n_peaks=400)
                    .locs_randomize()
                    .amps_constant(val=10000)
                )
                synth.CameraModel(bias=true_bg_mean, std=_true_bg_std)
                peaks.widths_uniform(_true_psf_std)
                if howmanydims==3:
                    ims = s.render_flchcy()
                elif howmanydims == 2:
                    ims = s.render_chcy()
                else:
                    ims = s.render_chcy()[0,0]
            return ims

        def it_can_calibrate_psf_uniform_im():
            #CREATE image with known PSF std
            true_psf_std = 0.5
            img = _synth_psf(true_bg_std,true_psf_std,howmanydims=1)
            #ESTIMATE locs and psfs for this image
            est_bg_mean, est_bg_std = worker._background_estimate(img, divs)
            im_sub = worker._background_subtraction(img, est_bg_mean)
            locs, reg_psfs = worker._calibrate_psf_im(im_sub, divs=divs, peak_mea=peak_mea)
            #CALCULATE psfs using fit_gauss2()
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
            #COMPARE fitted parameters to targets
            fit_params_mean = fit_params_sum / ((divs * divs) - blank_regions)
            true_params["std_x"]["tgt"] = true_psf_std
            true_params["std_y"]["tgt"] = true_psf_std
            _compare_fit_params(true_params, fit_params_mean)

        def it_can_calibrate_psf_uniform_im_w_large_psf_std():
            #CREATE image with known PSF std, much larger than in previous test
            true_psf_std = 2.5
            img = _synth_psf(true_bg_std,true_psf_std,howmanydims=1)
            #ESTIMATE locs and psfs for this image
            est_bg_mean, est_bg_std = worker._background_estimate(img, divs)
            im_sub = worker._background_subtraction(img, est_bg_mean)
            locs, reg_psfs = worker._calibrate_psf_im(im_sub, divs=divs, peak_mea=peak_mea)
            #CALCULATE psfs using fit_gauss2()
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
            #COMPARE fitted parameters to targets
            fit_params_mean = fit_params_sum / ((divs * divs) - blank_regions)
            true_params["std_x"]["tgt"] = true_psf_std
            true_params["std_y"]["tgt"] = true_psf_std
            _compare_fit_params(true_params, fit_params_mean)
        
        def it_can_calibrate_psf_im_nonuniform():
            #CREATE image with known small PSF std
            true_sm_psf_std = 0.5
            img1 = _synth_psf(true_bg_std,true_sm_psf_std,howmanydims=1)
            est_bg_mean, est_bg_std = worker._background_estimate(img1, divs)
            im_most = worker._background_subtraction(img1, est_bg_mean)
            #CREATE second image with known large PSF std
            true_lg_psf_std = 2.5
            img2 = _synth_psf(true_bg_std,true_lg_psf_std,howmanydims=1)
            est_bg_mean, est_bg_std = worker._background_estimate(img2, divs)
            im_corner = worker._background_subtraction(img2, est_bg_mean)
            #CREATE hybrid with large PSF std in one corner
            corner_divs = 2
            corner_size = corner_divs*int(im_most.shape[0]/divs)
            ys, xs = np.meshgrid(np.arange(im_most.shape[0]), np.arange(im_most.shape[1]))
            dist_to_corner = np.maximum(xs, ys)
            im3 = np.where(dist_to_corner < corner_size,im_corner,im_most)
            #VERIFY that at least 1 loc is within our corner with higher std,
            # otherwise some of the later code will error out anyway and this
            # should make the problem easier to debug
            locs, reg_psfs = worker._calibrate_psf_im(
                im3, divs=divs, peak_mea=peak_mea
            )
            nbr_in_test_corner_field = 0
            for x, y in locs:
                if (x < corner_size) and (y < corner_size):
                    nbr_in_test_corner_field += 1
            assert nbr_in_test_corner_field > 0
            #VERIFY that our params for "most" of the image, and the other
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
                if (x < corner_divs) and (y < corner_divs):
                    divisor_test_corner += 1
                    fit_params_sum_test_corner += np.array(fit_params)
                else:
                    divisor_most += 1
                    fit_params_sum_most += np.array(fit_params)
            assert ((divs * divs) - divisor_most - divisor_test_corner) <= 2
            fit_params_mean_most = fit_params_sum_most / divisor_most
            fit_params_mean_test_corner = fit_params_sum_test_corner / divisor_test_corner
            true_params["std_x"]["tgt"] = true_sm_psf_std
            true_params["std_y"]["tgt"] = true_sm_psf_std
            true_params["std_x"]["range"] = 1
            true_params["std_y"]["range"] = 1
            _compare_fit_params(true_params, fit_params_mean_most)
            true_params["std_x"]["tgt"] = true_lg_psf_std
            true_params["std_y"]["tgt"] = true_lg_psf_std
            _compare_fit_params(true_params, fit_params_mean_test_corner)

        zest()

    def it_can_calib_psf_stats_one_channel_one_cycle():
        #CREATE a set of peaks which will appear in all images
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
        synth.CameraModel(bias=bg_mean, std=bg_std)
        #GENERATE a z-stack of images with the same peaks, different psf stds
        templist = []
        center_z = ceil((n_z_slices - 1) / 2)
        # e.g. if n_z_slices=3, z's are [0,1,2], center_z = 1
        # idea here is to mimic having one z-slice which is most
        # focused by having its std be smaller, and then z-slices
        # further away from it have higher std
        std_min = 0.5
        std_max = std_min
        for z_i in range(0, n_z_slices):
            std_used = std_min * (1 + abs(center_z - z_i))
            std_max = max(std_used,std_max)
            peaks.widths_uniform(std_used)
            imgs = s.render_chcy()
            templist.append(imgs[0][0])
        z_stack = np.array([templist])
        #MEASURE the psf stats using this z-stack
        z_and_region_to_psf = np.array(
            worker._calib_psf_stats_one_channel(
                z_stack, n_fields, n_z_slices, calib, divs, peak_dim
            )
        )
        #DETERMINE if we measured things successfully
        fit_params_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        divisor = 0
        for x, y in grid_walk(divs):
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
        # TODO: find a more reasoned range for std
        std_range = 0.5 * (std_max - std_min)
        true_params["std_x"]["tgt"] = std_min
        true_params["std_x"]["range"] = std_range
        true_params["std_y"]["tgt"] = std_min
        true_params["std_y"]["range"] = std_range
        _compare_fit_params(true_params, fit_params_mean)

    def it_can_calibrate_foreground_one_channel():
        #MOCK a z_stack ImsImportResult
        calib = Calibration()
        n_z_slices = 15
        n_fields = 2
        ch_i = 0
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        ims = ims_import_result.ims[:, :, :]
        #CREATE a calib object already partly made
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
        #FIND radmat and locs
        fl_radmat, fl_loc = worker._calibrate_foreground_one_channel(
            calib, ims_import_result, n_fields, ch_i, sigproc_params
        )
        assert fl_radmat.shape[0] == fl_loc.shape[0] > 100 #found at least 100 locs
        assert fl_radmat.shape[1] == 1
        assert fl_radmat.shape[2] == n_z_slices
        assert fl_radmat.shape[3] == fl_loc.shape[1] == n_fields

        def it_can_calibrate_filter_locs_by_snr():
            #REUSE radmat and locs from previous test
            sig, locs = worker._calibrate_filter_locs_by_snr(
                fl_radmat, fl_loc, n_z_slices, ch_i
            )
            assert sig.shape[0] == locs.shape[0]
            assert fl_radmat.shape[0] == fl_loc.shape[0]
            assert (
                sig.shape[0] < fl_radmat.shape[0] * n_z_slices
            )  # i.e. it filtered something

            def it_can_calibrate_balance_calc():
                #REUSE ImsImportResult and filtered sig, locs from previous test
                balance = worker._calibrate_balance_calc(
                    ims_import_result, divs, sig, locs
                )
                assert np.min(balance) == 1  # the brightest area balanced at value 1
                assert np.count_nonzero(
                    balance == 1
                )  # i.e. only single spot is brightest

            zest()

        zest()

    def it_can_calibrate_psfs():
        # other tests for the lower level functions called by calibrate_psf()
        # are checking for the ability to return accurate measures of std,
        # handle nonuniform images, etc.  This test is just checking that
        # the data received is getting put into the calibration object in
        # the correct shape and with the correct keyword
        #CREATE mock calib and ims_import_result objects
        calib = Calibration()
        n_z_slices = 20
        n_fields = 3
        ims_import_result = result_from_z_stack(n_fields=n_fields, n_cycles=n_z_slices)
        calib = worker.calibrate_psf(calib, ims_import_result, divs, peak_mea)
        #VERIFY that the calib object got the right kinds of things added to it
        check.list_t(calib["regional_psf_zstack.instrument_channel[0]"], list)
        rbm_arr = np.array(calib["regional_psf_zstack.instrument_channel[0]"])
        check.array_t(
            rbm_arr, shape=(None, divs, divs, peak_mea, peak_mea), dtype=np.float64
        )

    def it_can_calibrate_regional_illumination_balance():
        #CREATE calib and ImsImportResult 
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
        #VERIFY that calibrate_regional_illumination_balance() puts data of the right
        # type and shape into the right place in the calib object.  Other tests will
        # test the lower level functions.
        check.list_t(calib["regional_illumination_balance.instrument_channel[0]"], list)
        rib_arr = np.array(calib["regional_illumination_balance.instrument_channel[0]"])
        check.array_t(rib_arr, shape=(divs, divs), dtype=np.float64)

    zest()

