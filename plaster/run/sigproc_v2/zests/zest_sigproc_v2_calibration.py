import numpy as np
from zest import zest
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.log.log import debug


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
