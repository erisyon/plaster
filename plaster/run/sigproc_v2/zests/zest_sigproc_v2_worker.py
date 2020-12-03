from munch import Munch
import numpy as np
import pickle
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.reg_psf import RegPSF
from plaster.run.sigproc_v2.sigproc_v2_task import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.log.log import debug
from plaster.tools.utils.tmp import tmp_folder
from zest import zest


def zest_sigproc_v2_worker_analyze():
    """
    Test the whole sigproc_v2 stack from top to bottom
    """

    def _run(reg_psf, s, extra_params=None):
        if extra_params is None:
            extra_params = {}

        sigproc_v2_params = SigprocV2Params(
            divs=5, peak_mea=13, calibration_file="", mode="analyze", **extra_params,
        )
        calib = Calibration()
        calib[f"regional_psf.instrument_channel[0]"] = reg_psf
        calib[f"regional_illumination_balance.instrument_channel[0]"] = np.ones((5, 5))

        ims_import_result = synth.synth_to_ims_import_result(s)
        # np.save("/erisyon/internal/_test.npy", ims_import_result.ims[0, 0])
        sigproc_v2_result = worker.sigproc_analyze(
            sigproc_v2_params, ims_import_result, progress=None, calib=calib
        )
        sigproc_v2_result.save()
        return sigproc_v2_result

    def it_returns_exact_distance_no_noise_one_cycle_one_peak():
        """
        Proves that peak finder is picking up the correct locations of sub-pixel positions
        """
        from scipy.spatial.distance import cdist  # Defer slow import

        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=1, overwrite=True, dim=(512, 512)
            ) as s:
                # Change the PSF in the corner to ensure that it is picking up the PSF
                reg_psf = RegPSF.fixture()

                peaks = synth.PeaksModelPSF(reg_psf, n_peaks=1).dyt_amp_constant(5000)
                peaks.locs = np.array([[100.3, 72.9]])

                sigproc_v2_result = _run(reg_psf, s)

                dists = cdist(peaks.locs, sigproc_v2_result.locs(), "euclidean")
                closest_iz = np.argmin(dists, axis=1)
                dists = dists[np.arange(dists.shape[0]), closest_iz]
                debug(dists)
                assert np.all(dists < 0.05)

    def it_returns_exact_sig_from_no_noise_no_collisions_no_bg_subtract():
        """
        Proves that the radiometry is exactly recovering many peaks with
        sub-pixel locations.
        """
        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=1, overwrite=True, dim=(512, 512)
            ) as s:
                reg_psf = RegPSF.fixture()

                (
                    synth.PeaksModelPSF(reg_psf, n_peaks=100)
                    .locs_grid(pad=50)
                    .amps_constant(5000)
                    .locs_add_random_subpixel()
                )

                sigproc_v2_result = _run(reg_psf, s, dict(low_inflection=-10.0))

                sig = sigproc_v2_result.sig()[:, 0, 0]
                assert np.all(np.abs(sig - 5000) < 0.75)

    def it_returns_exact_sig_from_no_noise_no_collisions_with_bg_subtract():
        """
        Proves that radiometry continues to get consistent answers when
        background subtraction is enabled.

        This is the same as it_returns_exact_sig_from_no_noise_no_collisions_no_bg_subtract()
        but WITH the bg subtraction turned on.

        It is expected that the background subtraction will lower the brightness of
        the peaks but do so consistently.
        """
        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=1, overwrite=True, dim=(512, 512)
            ) as s:
                reg_psf = RegPSF.fixture()

                (
                    synth.PeaksModelPSF(reg_psf, n_peaks=100)
                    .locs_grid(pad=50)
                    .amps_constant(5000)
                    .locs_add_random_subpixel()
                )

                sigproc_v2_result = _run(reg_psf, s)

                sig = sigproc_v2_result.sig()[:, 0, 0]
                # Background subtraction is expected to bring down the mean a little bit
                # In default settings it brings it to 4631
                assert np.all(np.abs(sig - 4631) < 4)

    def it_returns_good_signal_no_noise_multi_peak_multi_cycle():
        """
        Prove that the sub-pixel shifting of cycles does not substantially affect the
        quality of the radiometry.
        """

        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=3, overwrite=True, dim=(512, 512)
            ) as s:
                reg_psf = RegPSF.fixture()

                (
                    synth.PeaksModelPSF(reg_psf, n_peaks=800)
                    .amps_constant(5000)
                    .locs_grid(pad=40)
                    .locs_add_random_subpixel()
                )

                sigproc_v2_result = _run(reg_psf, s, dict(low_inflection=-10.0))

                sig = sigproc_v2_result.sig()[:, 0, :]

                # There is a small shift up to 5017
                assert np.all(np.abs(sig - 5017) < 3)

    def it_interpolates_regional_PSF_changes():
        """
        Prove that it handles a non-uniform PSF.

        Make a synth image with grid peak where we gradually shift the
        PSF across the image in both dimensions. Re-use the reg_psf
        for both the generation and the analysis.
        """
        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=1, overwrite=True, dim=(512, 512)
            ) as s:
                reg_psf = RegPSF.fixture_variable()

                (
                    synth.PeaksModelPSF(reg_psf, n_peaks=500)
                    .amps_constant(5000)
                    .locs_grid()
                    .locs_add_random_subpixel()
                )

                sigproc_v2_result = _run(reg_psf, s, dict(low_inflection=-10.0))

                sig = sigproc_v2_result.sig()[:, 0, :]

                # The reason this isn't more accurate is because in my radiometry C code
                # I don't have a good way to interpolate the lookup of the PSF
                # TODO: Fix the interpolation in C and revise this test
                assert np.std(sig) < 250
                assert 4900.0 < np.mean(sig) < 5100.0

    def it_returns_perfect_sig_on_uniform_psf_with_focus():
        """
        Prove that the focus correction works basically
        """
        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=3, overwrite=True, dim=(512, 512)
            ) as s:
                reg_psf = RegPSF.fixture()

                (
                    synth.PeaksModelPSF(
                        reg_psf, n_peaks=500, focus_per_cycle=[1.0, 0.90, 1.1]
                    )
                    .amps_constant(5000)
                    .locs_grid(pad=20)
                    .locs_add_random_subpixel()
                )

                sigproc_v2_result = _run(reg_psf, s, dict(low_inflection=-10.0))

                sig = sigproc_v2_result.sig()[:, 0, :]
                # np.save("/erisyon/internal/_sig.npy", sig)

                for cy_i in range(s.n_cycles):
                    assert 4950 < np.mean(sig[:, cy_i]) < 5100
                    assert np.std(sig[:, cy_i]) < 1.0

    def it_corrects_for_cycle_focal_changes_with_variable_PSF():
        """
        Prove that fit sampling of the Gaussians adjusts the focus and returns
        a perfect signal without considering alignment
        """
        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=3, overwrite=True, dim=(512, 512)
            ) as s:
                reg_psf = RegPSF.fixture_variable()

                (
                    synth.PeaksModelPSF(
                        reg_psf, n_peaks=500, focus_per_cycle=[1.0, 0.90, 1.10]
                    )
                    .amps_constant(5000)
                    .locs_grid(pad=20)
                    .locs_add_random_subpixel()
                )

                sigproc_v2_result = _run(reg_psf, s, dict(low_inflection=-10.0))

                sig = sigproc_v2_result.sig()[:, 0, :]

                # TODO: Need interpolation in C for this to work well
                for cy_i in range(s.n_cycles):
                    assert 4950 < np.mean(sig[:, cy_i]) < 5100
                    assert np.std(sig[:, cy_i]) < 1.0

    def it_operates_sanely_with_noise_uniform_psf():
        """A realistic total test that will need to be forgiving of noise, collisions, etc."""
        with tmp_folder(chdir=True):
            with synth.Synth(
                n_channels=1, n_cycles=3, overwrite=True, dim=(512, 512)
            ) as s:
                reg_psf = RegPSF.fixture()

                (
                    synth.PeaksModelPSF(
                        reg_psf, n_peaks=500, focus_per_cycle=[1.0, 0.90, 1.10]
                    )
                    .amps_constant(5000)
                    .locs_grid(pad=20)
                    .locs_add_random_subpixel()
                )
                synth.CameraModel(100, 30)
                synth.HaloModel()

                sigproc_v2_result = _run(reg_psf, s)

                sig = sigproc_v2_result.sig()[:, 0, :]
                # np.save("/erisyon/internal/_sig.npy", sig)
                # locs = sigproc_v2_result.locs()
                # np.save("/erisyon/internal/_locs.npy", locs)
                assert np.percentile(sig, 20) > 4900.0

    def it_operates_sanely_with_noise_variable_psf():
        """"""
        raise NotImplementedError

    def it_operates_sanely_with_noise_variable_psf_random_locations():
        """"""
        raise NotImplementedError

    def it_detects_signal_drops_over_cycles():
        """"""
        raise NotImplementedError

    def it_operates_sanely_with_noise_variable_psf_random_locations_non_uniform_lighting():
        """"""
        raise NotImplementedError


    def it_measures_aspect_ratio():
        """It returns high aspect ratio for collisions under a controlled two-peak system as the peaks approach each other"""
        raise NotImplementedError

    def it_runs_fitter_if_requested():
        """It will run the fitter over every peak"""
        raise NotImplementedError

    def it_applies_regional_balance():
        """It corrects for non-uniform illumination"""
        raise NotImplementedError

    def it_finds_peaks_as_density_increases():
        """Characterize how density affects quality"""
        raise NotImplementedError

    zest()


def zest_sigproc_v2_worker_calibrate():
    def it_extracts_regional_illumination_balance():
        raise NotImplementedError

    def it_extracts_psf_regionally():
        """
        Also: skips near edges, contention, nans, darks, bad-aspect ratio, etc.
        """
        raise NotImplementedError

    zest()


@zest.skip(reason="un-skip once we have multi-channel working")
def zest_channel_weights():
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


@zest.skip(reason="Need a massive overhaul since refactor")
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
