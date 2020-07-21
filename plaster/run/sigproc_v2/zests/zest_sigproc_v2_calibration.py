from math import floor

import numpy as np
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.image import imops
from plaster.tools.log.log import debug
from plaster.tools.utils.utils import np_within
from zest import zest


def zest_sigproc_v2_calibration():
    def it_estimates_uniform_background_correctly():
        divs = 5
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
        bg_mean, bg_std = worker.background_estimate(im, divs)
        assert np_within(np.mean(bg_mean), tgt_mean, 1)
        assert np_within(np.mean(bg_std), tgt_std, 1)
        return True

    def it_subtracts_uniform_bg_mean_correctly():
        divs = 5
        tgt_mean = 100
        tgt_std = 10
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            im = s.render_chcy()[0, 0]
        bg_mean, bg_std = worker.background_estimate(im, divs)
        im_sub = worker.background_subtraction(im, bg_mean)
        new_mean, new_std = worker.background_estimate(im_sub, divs)
        assert np_within(np.mean(new_mean), 0, (1 / tgt_std))
        return True

    def it_adds_regional_bg_stats_to_calib_correctly():
        divs = 5
        tgt_mean = 100
        tgt_std = 10
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
        return True

    def it_estimates_nonuniform_bg_mean_correctly():
        divs = 5
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
        bg_mean, bg_std = worker.background_estimate(im, divs)
        for x in range(0, divs):
            for y in range(0, divs):
                if x in [0, divs - 1] or y in [0, divs - 1]:
                    assert np_within((tgt_mean * 0.5), bg_mean[x][y], tgt_std / 2)
                    assert np_within(tgt_std * 0.5, bg_std[x][y], tgt_std / 4)
                else:
                    assert np_within(tgt_mean, bg_mean[x][y], tgt_std)
                    assert np_within(tgt_std, bg_std[x][y], tgt_std / 2)
        return True

    def it_estimates_nonuniform_bg_std_correctly():
        divs = 5
        tgt_mean = 100
        tgt_std1 = 10
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std1)
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
        bg_mean, bg_std = worker.background_estimate(im3, divs)
        for x in range(0, divs):
            for y in range(0, divs):
                if x in [0, divs - 1] or y in [0, divs - 1]:
                    assert np_within(bg_std[x][y], tgt_std2, tgt_std2 / 2)
                    assert np_within(bg_mean[x][y], tgt_mean, tgt_std2)
                else:
                    assert np_within(bg_std[x][y], tgt_std1, tgt_std1 / 2)
                    assert np_within(bg_mean[x][y], tgt_mean, tgt_std1)
        return True

    def it_subtracts_nonuniform_bg_mean_correctly():
        divs = 15
        border_ratio = 5
        tgt_mean = 100
        tgt_std = 10
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
        bg_mean, bg_std = worker.background_estimate(im, divs)
        im_sub = worker.background_subtraction(im, bg_mean)
        new_mean, new_std = worker.background_estimate(im_sub, divs)
        assert np_within(np.mean(new_mean), 0, tgt_std)
        return True

    def it_can_calibrate_psf_uniform_im():
        divs = 5
        tgt_mean = 100
        tgt_std = 10
        peak_mea = 11

        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=100)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=tgt_mean, std=tgt_std)

        psf_std = 0.5
        peaks.widths_uniform(psf_std)
        imgs = s.render_chcy()
        bg_mean, bg_std = worker.background_estimate(imgs[0, 0], divs)
        im_sub = worker.background_subtraction(imgs[0, 0], bg_mean)

        locs, reg_psfs = worker._calibrate_psf_im(
            im_sub, divs=divs, keep_dist=8, peak_mea=11, locs=None
        )

        fit_params_mean = [0, 0, 0, 0, 0, 0, 0, 0]
        fit_params_divisor = 0
        for x in range(0, divs):
            for y in range(0, divs):
                if np.sum(reg_psfs[x, y]) == 0:
                    continue
                fit_params, fit_variance = imops.fit_gauss2(reg_psfs[x, y])
                for fv in fit_variance:
                    assert fv < 0.001
                fit_params_mean = [a + b for a, b in zip(fit_params_mean, fit_params)]
                fit_params_divisor += 1
        fit_params_mean = [a / fit_params_divisor for a in fit_params_mean]
        midpt = floor(peak_mea / 2)
        # true_params: 1, psf_std, psf_std, midpt, midpt, 0, 0, peak_mea

        assert abs(1 - fit_params_mean[0]) < 0.1
        assert abs(psf_std - fit_params_mean[1]) < 0.15
        assert abs(psf_std - fit_params_mean[2]) < 0.15
        assert abs(midpt - fit_params_mean[3]) < 0.05
        assert abs(midpt - fit_params_mean[4]) < 0.05
        assert abs(fit_params_mean[5]) < 0.01
        assert abs(fit_params_mean[6]) < 0.01
        assert abs(peak_mea - fit_params_mean[7]) < 0.1
        return True

    def it_can_calibrate_psf_uniform_im_w_large_psf_std():
        divs = 5
        tgt_mean = 100
        tgt_std = 10
        peak_mea = 11

        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=100)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=tgt_mean, std=tgt_std)

        psf_std = 2.5
        peaks.widths_uniform(psf_std)
        imgs = s.render_chcy()
        bg_mean, bg_std = worker.background_estimate(imgs[0, 0], divs)
        im_sub = worker.background_subtraction(imgs[0, 0], bg_mean)

        locs, reg_psfs = worker._calibrate_psf_im(
            im_sub, divs=divs, keep_dist=8, peak_mea=11, locs=None
        )

        fit_params_mean = [0, 0, 0, 0, 0, 0, 0, 0]
        fit_params_divisor = 0
        for x in range(0, divs):
            for y in range(0, divs):
                if np.sum(reg_psfs[x, y]) == 0:
                    continue
                fit_params, fit_variance = imops.fit_gauss2(reg_psfs[x, y])
                for fv in fit_variance:
                    assert fv < 0.075
                fit_params_mean = [a + b for a, b in zip(fit_params_mean, fit_params)]
                fit_params_divisor += 1
        fit_params_mean = [a / fit_params_divisor for a in fit_params_mean]
        midpt = floor(peak_mea / 2)
        # true_params: 1, psf_std, psf_std, midpt, midpt, 0, 0, peak_mea
        assert abs(1 - fit_params_mean[0]) < 0.1
        assert abs(psf_std - fit_params_mean[1]) < 0.15
        assert abs(psf_std - fit_params_mean[2]) < 0.15
        assert abs(midpt - fit_params_mean[3]) < 0.05
        assert abs(midpt - fit_params_mean[4]) < 0.05
        assert abs(fit_params_mean[5]) < 0.01
        assert abs(fit_params_mean[6]) < 0.01
        assert abs(peak_mea - fit_params_mean[7]) < 0.1
        return True

    def it_can_calibrate_psf_im_nonuniform():

        divs = 5
        tgt_mean = 100
        tgt_std = 10
        peak_mea = 11
        n_z_slices = 2
        z_stack = np.array([])

        s = synth.Synth(n_cycles=1, overwrite=True)
        peaks = (
            synth.PeaksModelGaussianCircular(n_peaks=100)
            .locs_randomize()
            .amps_constant(val=10000)
        )
        synth.CameraModel(bias=tgt_mean, std=tgt_std)

        templist = []
        std_most = None
        std_corner = None
        for z_i in range(0, n_z_slices):
            std_used = 0.5 * (2 + z_i)
            if not std_most:
                std_most = std_used
            elif not std_corner:
                std_corner = std_used
            peaks.widths_uniform(std_used)
            imgs = s.render_chcy()
            bg_mean, bg_std = worker.background_estimate(imgs[0, 0], divs)
            im_sub = worker.background_subtraction(imgs[0, 0], bg_mean)
            templist.append(im_sub)
        z_stack = np.array(templist)
        # here we make an image w nonuniform std by using data from last image
        # in z_stack to replace data of 0th image, but only in one corner
        fields_in_corner = 2
        corner_xlim = fields_in_corner * z_stack[0].shape[0] / divs
        corner_ylim = fields_in_corner * z_stack[0].shape[1] / divs
        for x in range(0, z_stack[0].shape[0]):
            for y in range(0, z_stack[0].shape[1]):
                if (x < corner_xlim) and (y < corner_ylim):
                    z_stack[0][x][y] = z_stack[n_z_slices - 1][x][y]
        # verify that at least 1 loc is within our corner with higher std,
        # otherwise some of the later code will error out anyway and this
        # should make the problem easier to debug
        locs, reg_psfs = worker._calibrate_psf_im(
            z_stack[0], divs=divs, keep_dist=8, peak_mea=11, locs=None
        )
        nbr_in_corner_field = 0
        for x, y in locs:
            if (x < corner_xlim) and (y < corner_ylim):
                nbr_in_corner_field += 1
        assert nbr_in_corner_field > 0
        # see if our params for "most" of the image, and the other
        # part in the corner (with higher std) are close to the true answer
        fit_params_mean_most = [0, 0, 0, 0, 0, 0, 0, 0]
        fit_params_mean_corner = [0, 0, 0, 0, 0, 0, 0, 0]
        fit_params_divisor_most = 0
        fit_params_divisor_corner = 0
        for x in range(0, divs):
            for y in range(0, divs):
                if np.sum(reg_psfs[x, y]) == 0:
                    continue  # cannot use imops.fit_gauss2 on all-zero psf
                fit_params, fit_variance = imops.fit_gauss2(reg_psfs[x, y])
                # fit variance is a measure of how well fit_gauss2() thinks
                # it did in fitting to this psf; should be close to zero, i.e.
                # rather confident
                for fv in fit_variance:
                    assert fv < 0.0002
                if (x < fields_in_corner) and (y < fields_in_corner):
                    fit_params_divisor_corner += 1
                    fit_params_mean_corner = [
                        a + b for a, b in zip(fit_params_mean_corner, fit_params)
                    ]
                else:
                    fit_params_divisor_most += 1
                    fit_params_mean_most = [
                        a + b for a, b in zip(fit_params_mean_most, fit_params)
                    ]
        fit_params_mean_most = [
            a / fit_params_divisor_most for a in fit_params_mean_most
        ]
        fit_params_mean_corner = [
            a / fit_params_divisor_corner for a in fit_params_mean_corner
        ]
        midpt = floor(peak_mea / 2)
        # true_params: 1, std_most, std_most, midpt, midpt, 0, 0, peak_mea
        assert abs(1 - fit_params_mean_most[0]) < 0.1
        assert abs(std_most - fit_params_mean_most[1]) < 0.15
        assert abs(std_most - fit_params_mean_most[2]) < 0.15
        assert abs(midpt - fit_params_mean_most[3]) < 0.05
        assert abs(midpt - fit_params_mean_most[4]) < 0.05
        assert abs(fit_params_mean_most[5]) < 0.025
        assert abs(fit_params_mean_most[6]) < 0.025
        assert abs(peak_mea - fit_params_mean_most[7]) < 0.1
        # true_params: 1, std_corner, std_corner, midpt, midpt, 0, 0, peak_mea
        assert abs(1 - fit_params_mean_corner[0]) < 0.1
        assert abs(std_corner - fit_params_mean_corner[1]) < 0.15
        assert abs(std_corner - fit_params_mean_corner[2]) < 0.15
        assert abs(midpt - fit_params_mean_corner[3]) < 0.05
        assert abs(midpt - fit_params_mean_corner[4]) < 0.05
        assert abs(fit_params_mean_corner[5]) < 0.05
        assert abs(fit_params_mean_corner[6]) < 0.05
        assert abs(peak_mea - fit_params_mean_corner[7]) < 0.1
        return True

    def it_can_do__calibrate():
        divs = 5
        tgt_mean = 100
        tgt_std = 10
        with synth.Synth(overwrite=True) as s:
            (
                synth.PeaksModelGaussianCircular(n_peaks=100)
                .locs_randomize()
                .amps_constant(val=10000)
            )
            synth.CameraModel(bias=tgt_mean, std=tgt_std)
            ims = s.render_flchcy()
        calib = worker._calibrate(ims, divs=5, progress=None, overload_psf=None)
        assert calib["regional_bg_mean.instrument_channel[0]"]
        assert calib["regional_bg_std.instrument_channel[0]"]
        assert calib["regional_psf_zstack.instrument_channel[0]"]
        return True

    zest()


# @zest.skip("n", "Not ready")
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
