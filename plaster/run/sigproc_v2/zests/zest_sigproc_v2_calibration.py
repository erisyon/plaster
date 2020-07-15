import numpy as np
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
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
        # assert abs(np.mean(new_mean)) < (1 / tgt_std)
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
