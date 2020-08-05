# @TODO this was brought over from archive pytest stuff for synth.  Synth works again,
# but these tests were not ported.
#
# This was begun to debug the multi-channel sigproc_v1 bug, but I went down a different road and
# started exploring that in explore/synth.ipynb so that I could viz/understand the bug.
# tfb 31 jan 2020
#


# import numpy as np
# import pandas as pd
# import math
# from zest import zest
# from scipy import stats
# from plaster.tools.utils.utils import np_array_same
# from scipy.spatial import distance
# from plaster.run.sigproc_v1.sigproc_params import SigprocParams
# from plaster.run.sigproc_v1.sigproc_result import SigprocResult
# from plaster.run.sigproc_v1.zests.synth import Synth
# from plaster.tools.image.coord import XY, YX, WH, HW, ROI
# from plaster.tools.log.log import debug
# from retrying import retry
#
# class FoundToTruthComparator:
#     """Compares found and true peaks"""
#     def __init__(self, pfp, synth):
#         pfp.peak_df["corr_x"] = pfp.peak_df.x - pfp.border_size
#         pfp.peak_df["corr_y"] = pfp.peak_df.y - pfp.border_size
#
#         self.found_locs = pfp.peak_df.apply(
#             lambda row: XY(row.corr_x, row.corr_y), axis=1
#         ).tolist()
#         self.truth_locs = synth.peak_locs
#
#         self.n_found = len(self.found_locs)
#         self.n_true = len(synth.peak_locs)
#
#         # found_locs are on the rows
#         # truth_locs are the cols
#         self.dist = distance.cdist(self.found_locs, self.truth_locs, "euclidean")
#
#         self.truth_i_to_found_i = self.dist.argmin(axis=0)
#         self.truth_to_found_dist = self.dist.min(axis=0)
#
#         self.found_i_to_truth_i = self.dist.argmin(axis=1)
#         self.found_to_truth_dist = self.dist.min(axis=1)
#
#         min_dist = 1.5
#
#         border = HW(pfp.border_size, pfp.border_size)
#         self.true_peaks_that_were_lost = [
#             (
#                 synth.peak_locs[i],
#                 synth.peak_locs[i] + border,
#                 self.truth_to_found_dist[i],
#                 synth.closest[i],
#             )
#             for i in range(self.n_true)
#             if self.truth_to_found_dist[i] > min_dist
#         ]
#
#         self.n_found_peaks_that_were_true = (self.truth_to_found_dist <= min_dist).sum()
#         self.n_true_peaks_that_were_lost = (
#             self.dist.shape[1] - self.n_found_peaks_that_were_true
#         )
#         self.n_found_peaks_that_were_not_true = (
#             self.found_to_truth_dist > min_dist
#         ).sum()
#
# def _assert_shifts_are_right(self, frame_df, synth):
#     assert np.all(
#         (frame_df.shift_x - synth.frame_xs) ** 2
#         + (frame_df.shift_y - synth.frame_ys) ** 2
#         <= 1.5 ** 2
#     )
#
# def _assert_radmat_is_right_shape(self, radmat, synth):
#     assert radmat.shape[1] == synth.n_channels
#     assert radmat.shape[2] == synth.n_cycles
#
# def _assert_areas(self, radmat, synth):
#     for cycle in range(synth.n_cycles):
#         peak_values = radmat[:, 0:1, cycle : cycle + 1]
#         _min = np.min(synth.peak_areas[:, cycle, :]) * 0.95
#         _max = np.max(synth.peak_areas[:, cycle, :]) * 1.05
#
#         if not (all(_min < peak_values) and all(peak_values < _max)):
#             debug(_min)
#             debug(_max)
#             debug(peak_values)
#
#         assert all(_min < peak_values) and all(peak_values < _max)
#
# def _peak_finder( synth, debug_tag=None, **kwargs):
#     if kwargs.get("hat_rad") is None:
#         kwargs["hat_rad"] = 3
#
#     if kwargs.get("anomaly_iqr_cutoff") is None:
#         # When I added the anomaly_iqr_cutoff that broke a lot of tests
#         # that didn't have power of 2 dimensions so this is here
#         # to turn off anomaly_iqr_cutoff on those tests
#         kwargs["anomaly_iqr_cutoff"] = None
#
#     # peak_symmetry_cutoff = 1000.0,
#     # channel_indices_for_alignment = list(range(synth.n_channels)),
#
#     sigproc_params = SigprocParams(**kwargs)
#     sigproc_params.set_radiometry_channels_from_input_channels_if_needed(
#         synth.n_channels
#     )
#
#     ims_import_result = ims_result_klass.load_from_folder(self.inputs.ims_import)
#
#         # TASK: Figure out a better solution for this.
#         sigproc_params.set_radiometry_channels_from_input_channels_if_needed(
#             ims_import_result.n_channels
#         )
#
#         sigproc_result = sigproc_v1(sigproc_params, ims_import_result, self.progress)
#
#         sigproc_result.save()
#
#
#
#
#     sigproc_result = Sigproc(
#         synth.field_stack(),
#         0,
#         SigprocParams(
#             peak_symmetry_cutoff=1000.0,
#             channel_indices_for_alignment=list(range(synth.n_channels)),
#             radiometry_channels={str(i): i for i in range(synth.n_channels)},
#             **kwargs,
#         ),
#     )
#     spp.save(f"/tmp/{debug_tag}.ipkl")
#     spp.save_debug(f"/tmp/_{debug_tag}.ipkl")
#     return spp
#
#
# def zest_sigproc_synth():
#
#     def _before():
#         print( "\n_before")
#         np.random.seed(100)
#
#     def it_aligns_channels_independently():
#         print( "\nit_aligns_channels_independently")
#         synth = Synth(
#             n_cycles=3,
#             n_peaks=3,
#             n_channels=2,
#             dim=WH(50, 50),
#             bg_std=0.2,
#             peak_focus=1.0,
#             peak_area_by_channel_cycle=np.array(
#                 [
#                     [  # Channel 0
#                         [6, 6, 6],  # Cycle 0  # Peak 0  # Peak 1  # Peak 2
#                         [6, 6, 6],  # Cycle 1  # Peak 0  # Peak 1  # Peak 2
#                         [6, 6, 6],  # Cycle 2  # Peak 0  # Peak 1  # Peak 2
#                     ],
#                     [  # Channel 1
#                         [30, 30, 30],  # Cycle 0  # Peak 0  # Peak 1  # Peak 2
#                         [30, 30, 30],  # Cycle 1  # Peak 0  # Peak 1  # Peak 2
#                         [20, 20, 30],  # Cycle 2  # Peak 0  # Peak 1  # Peak 2
#                     ],
#                 ]
#             ),
#             peak_xs=(20, 30, 10),
#             peak_ys=(23, 35, 30),
#         )
#
#         debug_tag = "test_it_aligns_channels_independently"
#         pfp = _peak_finder(synth, debug_tag=debug_tag)
#
#         df_cycles = pfp.frame_df.groupby("cycle")
#         for cycle, group in df_cycles:
#             assert len(group.shift_x.unique()) == 1
#             assert len(group.shift_y.unique()) == 1
#
#         compare = FoundToTruthComparator(pfp, synth)
#         assert compare.n_found_peaks_that_were_true == synth.n_peaks
#         assert compare.n_true_peaks_that_were_lost == 0
#         assert compare.n_found_peaks_that_were_not_true == 0
#
#         zest()
#
#     # def it_finds_one_peak_in_low_noise():
#     #     """This is a simple sanity check. If this fails, something has gone seriously wrong"""
#     #     synth = Synth(
#     #         n_cycles=1,
#     #         n_peaks=1,
#     #         n_channels=1,
#     #         dim=WH(50, 50),
#     #         bg_std=0.01,
#     #         peak_focus=1.0,
#     #         peak_area_all_cycles=(60,),
#     #         peak_xs=(25,),
#     #         peak_ys=(25,),
#     #     )
#     #
#     #     debug_tag = "test_it_finds_one_peak_in_low_noise"
#     #     spp = self._peak_finder(synth, debug_tag=debug_tag)
#     #
#     #     n_frames = len(spp.frame_df.index)
#     #     assert n_frames == 1
#     #
#     #     n_peaks = len(spp.peak_df.index)
#     #     assert n_peaks == 1
#     #     assert spp.signal_radmat.shape == (1, 1, 1)
#     #
#     #     _assert_shifts_are_right(spp.frame_df, synth)
#     #     _assert_radmat_is_right_shape(spp.signal_radmat, synth)
#     #     _assert_radmat_is_right_shape(spp.noise_radmat, synth)
#     #     _assert_areas(spp.signal_radmat, synth)
#     #
#     #     compare = FoundToTruthComparator(spp, synth)
#     #     assert compare.n_found_peaks_that_were_true == synth.n_peaks
#     #     assert compare.n_true_peaks_that_were_lost == 0
#     #     assert compare.n_found_peaks_that_were_not_true == 0
#     #     assert compare.found_locs == [XY(25, 25)]
#     #
#     # def it_finds_grid_peaks_with_square_wave_with_low_noise():
#     #     """This is another sanity check. If this fails something has gone seriously wrong"""
#     #     synth = Synth(
#     #         n_cycles=4,
#     #         n_peaks=1,
#     #         n_channels=1,
#     #         dim=WH(50, 50),
#     #         bg_std=0.00,
#     #         peak_focus=1.0,
#     #         peak_area_all_cycles=(30, 60, 30, 60),
#     #         grid_distribution=True,
#     #     )
#     #
#     #     debug_tag = "test_it_finds_grid_peaks_with_square_wave_with_low_noise"
#     #     pfp = self._peak_finder(synth, debug_tag=debug_tag)
#     #     self._assert_areas(pfp.signal_radmat, synth)
#     #
#     #     compare = FoundToTruthComparator(pfp, synth)
#     #     assert compare.n_found_peaks_that_were_true == synth.n_peaks
#     #     assert compare.n_true_peaks_that_were_lost == 0
#     #     assert compare.n_found_peaks_that_were_not_true == 0
#     #
#     # def characterize_the_positional_offsets_no_noise():
#     #     """Move a single peak around"""
#     #     samples = []
#     #     for x in np.linspace(
#     #         -1.0, 1.0, 7
#     #     ):  # An odd step size here so that the center pixel will hit exactly
#     #         synth = Synth(
#     #             n_cycles=1,
#     #             n_peaks=1,
#     #             n_channels=1,
#     #             dim=WH(50, 50),
#     #             bg_std=0.0,
#     #             peak_focus=1.0,
#     #             peak_area_all_cycles=(60,),
#     #             # peak_mean=(60,),
#     #             # peak_std=0,
#     #             peak_xs=(25 + x,),
#     #             peak_ys=(25,),
#     #         )
#     #
#     #         debug_tag = f"test_characterize_the_positional_offsets_no_noise-{x:.1f}"
#     #         pfp = self._peak_finder(synth, debug_tag=debug_tag)
#     #         assert pfp.n_peaks == 1
#     #         samples += [pfp.signal_radmat[0, 0, 0]]
#     #
#     #     samples = np.array(samples)
#     #     n_samples = samples.shape[0]
#     #     _min = np.min(samples)
#     #     _max = np.max(samples)
#     #
#     #     # The center point should have the maximum value (other places also allowed to have max)
#     #     assert samples[n_samples // 2] == _max
#     #
#     #     # The min and max shouldn't vary by more than 3%
#     #     assert _min / _max > 0.97
#     #
#     #     # When the peak_focus above is 1.0 there shouldn't be any loss out of the window
#     #     assert _min > 59.5
#     #
#     # def characterize_approach_distance():
#     #     lost_at_dist = None
#     #     for x in np.linspace(10, 0.0, 11):
#     #         synth = Synth(
#     #             n_cycles=1,
#     #             n_peaks=2,
#     #             n_channels=1,
#     #             dim=WH(50, 50),
#     #             bg_std=0.0,
#     #             peak_focus=1.0,
#     #             peak_mean=(60,),
#     #             peak_std=0,
#     #             peak_xs=(25, 25 + x),
#     #             peak_ys=(25, 25),
#     #         )
#     #
#     #         debug_tag = f"test_characterize_approach_distance-{x:.1f}"
#     #         pfp = self._peak_finder(synth, debug_tag=debug_tag)
#     #         if pfp.signal_radmat.shape[0] == 1 and lost_at_dist is None:
#     #             lost_at_dist = x
#     #             break
#     #
#     #     assert lost_at_dist <= 5
#     #
#     # @retry(stop_max_attempt_number=3)
#     # def characterize_recall_as_density_increases():
#     #     n_peak_samples = 20
#     #     n_trials_per_peak = 4
#     #     found = []
#     #     for n_peaks in range(1, n_peak_samples, 4):
#     #         for trial in range(n_trials_per_peak):
#     #             synth = Synth(
#     #                 n_cycles=1,
#     #                 n_peaks=n_peaks,
#     #                 n_channels=1,
#     #                 dim=WH(50, 50),
#     #                 bg_std=0.0,
#     #                 peak_focus=1.0,
#     #                 peak_mean=(60,),
#     #                 peak_std=0,
#     #             )
#     #             pfp = _peak_finder(
#     #                 synth,
#     #                 iqr_rng=None,
#     #                 debug_tag=f"as_density_increases_{n_peaks}_{trial}",
#     #             )
#     #             found += [(n_peaks, pfp.signal_radmat.shape[0])]
#     #
#     #     found_df = pd.DataFrame(found, columns=["actual", "found"])
#     #     m = found_df.groupby("actual", as_index=False).mean()
#     #     staturation = 17
#     #
#     #     def lin_fit(which):
#     #         of_interest = m.loc[which]
#     #         xs = of_interest.actual.values
#     #         ys = of_interest.found.values
#     #         slope, intercept, _, _, std_err = stats.linregress(xs, ys)
#     #         return slope, std_err
#     #
#     #     # The found up to a density up to saturation should fit a line well with a slope ~0.5
#     #     # debug(m.actual)
#     #     # debug(m.found)
#     #     slope, std_err = lin_fit(m.actual < staturation)
#     #     assert 0.4 < slope < 0.92 and std_err < 0.2
#     #
#     #     # The found points past the saturation should fit a line well with a slope ~0
#     #     # slope, std_err = lin_fit(m.actual >= staturation)
#     #     # debug(slope, std_err)
#     #     # assert -0.3 < slope < 0.1 and std_err < 0.1
#     #
#     # def characterize_detect_in_noise_single_cycle():
#     #     sigs = []
#     #     nois = []
#     #     bgs = []
#     #     for bg_std in np.linspace(0.0, 1.5, 15):
#     #         synth = Synth(
#     #             n_cycles=1,
#     #             n_peaks=1,
#     #             n_channels=1,
#     #             dim=WH(50, 50),
#     #             bg_std=bg_std,
#     #             peak_focus=1.0,
#     #             peak_mean=(60,),
#     #             peak_std=0,
#     #             peak_xs=(25,),
#     #             peak_ys=(25,),
#     #         )
#     #
#     #         debug_tag = f"test_characterize_detect_in_noise_single_cycle-{bg_std:.1f}"
#     #         pfp = self._peak_finder(synth, debug_tag=debug_tag)
#     #         sigs += [pfp.signal_radmat[0, 0, 0]]
#     #         nois += [pfp.noise_radmat[0, 0, 0]]
#     #         bgs += [bg_std]
#     #
#     #     slope, intercept, _, _, std_err = stats.linregress(bgs, nois)
#     #     assert 2.0 < slope < 2.75
#     #     assert -0.1 < intercept < 0.42
#     #
#     # @zest.skip(reason="WIP")
#     # def characterize_peak_finding_as_focus_changes(self):
#     #     """As the focus gets worse, the peak finder should adjust its kernel"""
#     #
#     #     # Instead of implementing this right now, maybe I should pull in the
#     #     # gaussian fits from Alex's code and look at the distributions
#     #
#     #     in_focus_synth = Synth(
#     #         n_cycles=1,
#     #         n_peaks=1,
#     #         n_channels=1,
#     #         dim=WH(50, 50),
#     #         bg_std=0.0,
#     #         peak_focus=1.0,
#     #         peak_area_all_cycles=(50,),
#     #         peak_xs=(25,),
#     #         peak_ys=(25,),
#     #     )
#     #
#     #     out_focus_synth = Synth(
#     #         n_cycles=1,
#     #         n_peaks=1,
#     #         n_channels=1,
#     #         dim=WH(50, 50),
#     #         bg_std=0.0,
#     #         peak_focus=1.4,
#     #         peak_area_all_cycles=(50,),
#     #         peak_xs=(25,),
#     #         peak_ys=(25,),
#     #     )
#     #
#     #     np.save("_in_focus", in_focus_synth.ims)
#     #     np.save("_out_focus", out_focus_synth.ims)
#     #
#     # def characterize_peak_finding_as_it_slides_along_the_grid():
#     #     for x in np.linspace(0.0, 1.0, 20):
#     #         synth = Synth(
#     #             n_cycles=1,
#     #             n_peaks=1,
#     #             n_channels=1,
#     #             dim=WH(50, 50),
#     #             bg_std=0.0,
#     #             peak_focus=1.0,
#     #             peak_area_all_cycles=(50,),
#     #             peak_xs=(25 + x,),
#     #             peak_ys=(25,),
#     #         )
#     #         debug_tag = (
#     #             f"test_characterize_peak_finding_as_it_slides_along_the_grid-{x:.2f}"
#     #         )
#     #         pfp = _peak_finder(synth, debug_tag=debug_tag)
#     #         fore = pfp.signal_radmat[0, 0, 0]
#     #         assert fore > 49.5
#     #
#     # def characterize_peak_finding_near_edges():
#     #     for x in np.linspace(0.0, 30, 31):
#     #         synth = Synth(
#     #             n_cycles=1,
#     #             n_peaks=1,
#     #             n_channels=1,
#     #             dim=WH(30, 30),
#     #             bg_std=0.0,
#     #             peak_focus=1.0,
#     #             peak_area_all_cycles=(50,),
#     #             peak_xs=(x,),
#     #             peak_ys=(15,),
#     #         )
#     #         debug_tag = f"test_characterize_peak_finding_near_edges-{x:.1f}"
#     #         pfp = _peak_finder(synth, debug_tag=debug_tag)
#     #         fore = pfp.signal_radmat[:, 0, 0]
#     #         if len(fore) == 0:
#     #             # Not found
#     #             assert x <= 9 or x >= 22
#     #
#     # @zest.skip(reason="Symmetry code disabled")
#     # def test_characterize_symmetry_detection(self):
#     #     for a in np.linspace(0.0, math.pi / 4.0, 5):
#     #         for r in np.linspace(4.0, 0.0, 10):
#     #             synth = Synth(
#     #                 n_cycles=1,
#     #                 n_peaks=2,
#     #                 n_channels=1,
#     #                 dim=WH(50, 50),
#     #                 bg_std=0.0,
#     #                 peak_focus=1.0,
#     #                 peak_area_all_cycles=(50,),
#     #                 peak_xs=(25, 25 + r),
#     #                 peak_ys=(25, 25),
#     #             )
#     #             debug_tag = f"test_characterize_symmetry_detection-{r:.1f}-{a:.1f}"
#     #             pfp = self._peak_finder(synth, debug_tag=debug_tag)
#     #             if r >= 0.8:
#     #                 assert pfp.peak_df.symmetry[0] > 1.1
#     #             else:
#     #                 assert pfp.peak_df.symmetry[0] <= 1.2
#     #
#     # def characterize_snr_when_peak_is_misaligned(self):
#     #     for x in np.linspace(0.0, 1.0, 11):
#     #         synth = Synth(
#     #             n_cycles=1,
#     #             n_peaks=1,
#     #             n_channels=1,
#     #             dim=WH(50, 50),
#     #             bg_std=0.0,
#     #             peak_focus=1.0,
#     #             peak_area_all_cycles=(50,),
#     #             peak_xs=(25 + x,),
#     #             peak_ys=(25,),
#     #         )
#     #         debug_tag = f"test_characterize_snr_when_peak_is_misapligned-{x:.1f}"
#     #         pfp = _peak_finder(synth, debug_tag=debug_tag)
#     #
#     #         signal_gaussian = pfp.signal_radmat[0, 0, 0]
#     #         noise_gaussian = pfp.noise_radmat[0, 0, 0]
#     #         assert np.abs(50.0 - signal_gaussian) < 0.1
#     #         assert noise_gaussian < 0.1
#     #
#     # def it_discards_nan_rows():
#     #     synth = Synth(
#     #         n_cycles=2,
#     #         n_peaks=3,
#     #         n_channels=2,
#     #         dim=WH(50, 50),
#     #         bg_std=0.0,
#     #         peak_focus=1.0,
#     #         peak_area_all_cycles=(50, 50),
#     #         peak_xs=(10, 20, 30),
#     #         peak_ys=(25, 25, 25),
#     #     )
#     #     debug_tag = "test_it_discards_nan_rows"
#     #     pfp = _peak_finder(synth, debug_tag=debug_tag)
#     #     sig = np.ones(pfp.signal_radmat.shape)
#     #     noi = np.ones(pfp.noise_radmat.shape)
#     #     assert sig.shape == noi.shape
#     #     sig[1, 1, 1] = np.nan
#     #     noi[2, 0, 1] = np.nan
#     #     pfp._remove_nans_from_radiometry(sig, noi)
#     #     expected = np.array(
#     #         [
#     #             [[1.0, 1.0], [1.0, 1.0]],
#     #             [[0.0, 0.0], [0.0, 0.0]],
#     #             [[0.0, 0.0], [0.0, 0.0]],
#     #         ]
#     #     )
#     #     assert np_array_same(sig, expected) and np_array_same(noi, expected)
#     #
#     # def it_removes_anomalies(self):
#     #     synth = Synth(
#     #         n_cycles=2,
#     #         n_peaks=3,
#     #         n_channels=1,
#     #         dim=WH(4 * 15, 4 * 15),
#     #         bg_std=0.2,
#     #         peak_focus=1.0,
#     #         peak_area_by_channel_cycle=np.array(
#     #             [
#     #                 [  # Channel 0
#     #                     [50, 20, 20],  # Cycle 0  # Peak 0, 1, 2
#     #                     [50, 20, 20],  # Cycle 1  # Peak 0, 1, 2
#     #                 ]
#     #             ]
#     #         ),
#     #         peak_xs=(20, 30, 10),
#     #         peak_ys=(23, 35, 30),
#     #         frame_offsets=[XY(0, 0), XY(10, 8)],
#     #     )
#     #
#     #     # Place an anomaly in the corner of the second cycle.
#     #     # This will cause un-corrected ones to mis-align
#     #     synth.ims[0, 1, 50:60, 50:60] = 100
#     #
#     #     # Without anomaly_iqr_cutoff the alignment should get fooled
#     #     pfp = _peak_finder(
#     #         synth, debug_tag="no_anomaly_removal", anomaly_iqr_cutoff=None
#     #     )
#     #     assert pfp.alignment_offsets[1][0] > 20
#     #     assert pfp.peak_df.shape[0] == 1
#     #     assert pfp.raw_mask_rects[0] is None
#     #
#     #     # With anomaly_iqr_cutoff the bad part should get masked out
#     #     pfp = _peak_finder(
#     #         synth, debug_tag="with_anomaly_removal", anomaly_iqr_cutoff=95
#     #     )
#     #     assert pfp.alignment_offsets[1] == XY(10, 8)
#     #     assert pfp.peak_df.shape[0] == 2
#     #     assert len(pfp.raw_mask_rects) == 1  # 1 channel
#     #     assert len(pfp.raw_mask_rects[0]) == 2  # 2 cycles
#     #     assert (
#     #         len(pfp.raw_mask_rects[0][1]) == 1
#     #     )  # Second cycle should have a mask rect
#     #
#     # def it_excludes_peaks_from_anomalies():
#     #     synth = Synth(
#     #         n_cycles=2,
#     #         n_peaks=3,
#     #         n_channels=1,
#     #         dim=WH(4 * 30, 4 * 30),
#     #         bg_std=0.2,
#     #         peak_focus=1.0,
#     #         peak_area_by_channel_cycle=np.array(
#     #             [
#     #                 [  # Channel 0
#     #                     [20, 20, 20],  # Cycle 0  # Peak 0, 1,
#     #                     [20, 20, 20],  # Cycle 1  # Peak 0, 1,
#     #                 ]
#     #             ]
#     #         ),
#     #         peak_xs=(10, 40, 100),
#     #         peak_ys=(30, 30, 100),
#     #         frame_offsets=[XY(0, 0), XY(0, 0)],
#     #     )
#     #
#     #     # Place an anomaly in the corner of the second cycle.
#     #     # This will cause un-corrected ones to mis-align
#     #     synth.ims[0, 1, 25:35, 30:50] += 0.50
#     #     pfp = _peak_finder(
#     #         synth,
#     #         debug_tag="excludes_peaks_from_anomalies",
#     #         anomaly_iqr_cutoff=97,
#     #         threshold_abs=0.5,
#     #     )
#     #     assert pfp.peak_df.shape[0] == 2
#     #     assert pfp.peak_df.loc[0].x == 100.0 and pfp.peak_df.loc[0].y == 100.0
#     #     assert pfp.peak_df.loc[1].x == 10.0 and pfp.peak_df.loc[1].y == 30.0
#
#
#     zest()
