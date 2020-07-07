import itertools
from munch import Munch
import numpy as np
from plaster.tools.utils import data
from plaster.tools.schema import check
from plaster.tools.parallel_map.parallel_map import parallel_array_split_map
from plaster.tools.log.log import debug


def calib_nn(calib_nn_params, sigproc_result, progress, pipeline):
    pass


'''


def ac_calib_fit_gain(scope_radmat, phase_callback=None):
    assert scope_radmat.ndim == 3  # If not I need to change it
    n_samples, n_channels, n_cycles = scope_radmat.shape

    # FIT each channel
    per_channel_gains = [None] * n_channels
    per_channel_convergence_debugging = [None] * n_channels
    for ch in range(n_channels):

        def _phase_callback(i):
            if phase_callback is not None:
                return phase_callback(ch * 2 + i, n_channels * 2)

        X = np.array(scope_radmat[:, ch, :])
        gain, convergence_debugging = gain_fitter(X, phase_callback=_phase_callback)

        per_channel_gains[ch] = gain
        per_channel_convergence_debugging[ch] = convergence_debugging

    return per_channel_gains, per_channel_convergence_debugging


class CalibTask(PipelineTask):
    def start(self):
        sigproc_dir = local.path(self.inputs.sigproc_v1)
        scope_radmat = np.load(sigproc_dir / "radmat.npy")

        n_pres, n_mocks, n_edmans = (
            self.config.parameters.n_pres,
            self.config.parameters.n_mocks,
            self.config.parameters.n_edmans,
        )
        n_cycles = n_pres + n_mocks + n_edmans

        if self.config.parameters.mode == "gain":
            first_cycle = n_pres + n_mocks - 1
            assert first_cycle >= 0
            second_cycle = first_cycle + 2
            assert second_cycle < n_cycles

            channels = np.array(self.config.parameters.channels)
            scope_radmat = scope_radmat[:, channels, first_cycle:second_cycle]

            with ParallelMapContext(progress=self.progress):
                (
                    per_channel_gains,
                    per_channel_convergence_debugging,
                ) = ac_calib_fit_gain(scope_radmat, phase_callback=self.set_phase)

            utils.pickle_write(
                "calib_gain.pkl",
                ch_gains=per_channel_gains,
                ch_convergence_debugging=per_channel_convergence_debugging,
                first_cycle=first_cycle,
                second_cycle=second_cycle,
                channels=channels,
            )


@check.args
def vpd_fitter(uX, dt_ann: DyetracksANN, n_dyes: int, cheat_dt_iz=None):
    """
    Fit the "variance per dye" for one channel.

    Arguments:
        uX: An X matrix scaled by gain making it a "unity" scaled X (hence uX)
        dt_ann: DyetracksANN for channel
    """

    assert np.all(uX < 100.0)  # Sanity check that a unity X was passed in

    if cheat_dt_iz is not None:
        check.array_t(cheat_dt_iz, (uX.shape[0],))
        dt_iz = cheat_dt_iz
    else:
        dt_iz = dt_ann.get_iz(uX)

    P = dt_ann[dt_iz]

    dye_count_to_std = np.full((n_dyes,), np.nan)
    dye_count_to_n_samples = np.zeros((n_dyes,))

    for i in range(0, n_dyes):
        # MASK for all the mapped dye_tracks that have this dye_count
        mask = dt_iz == i

        # Get any element of X that has dye_count of i
        x = uX[mask]
        dye_count_to_std[i] = np.std(x)
        dye_count_to_n_samples[i] = mask.sum()

    # Fit an model. Log normal distribution has approximately linear std as func of the brightness
    def model(x, a):
        return a * x

    xy = np.array(
        [
            (i, dye_count_to_std[i])
            for i in range(n_dyes)
            if dye_count_to_n_samples[i] > 100
        ]
    )

    if len(xy) <= 4:
        important(
            f"There's not enough samples to determine the variance {dye_count_to_n_samples}"
        )
        # raise ValueError(
        #     f"There's not enough samples to determine the variance {dye_count_to_n_samples}"
        # )

    popt, pcov = curve_fit(model, xy[:, 0], xy[:, 1], p0=(1.0,))
    std_slope = popt[0]

    # Use the model to produce smooth values from std to var
    variance_per_dye = np.array([model(i, std_slope) for i in range(n_dyes)]) ** 2

    # The 0-th element is hard to estimate and will explode if zero
    # so we use the variance in the 1-st term.
    variance_per_dye[0] = variance_per_dye[1]
    return variance_per_dye, std_slope, dye_count_to_n_samples


def ac_calib_vpd(scope_radmat, train_y, train_dyetracks, phase_callback=None):
    for ch in range(n_channels):
        X = np.array(radmat[:, ch, :])
        ch_dt_ann = DyetracksANN(
            utils.np_flatten_all_but_first(train_data_dye_tracks[:, ch, :])
        )

        gain = gains[ch]
        channel_gains[ch] = gain

        channel_uX = X / gain

        # Build inverse covariances. All of this is unnecessarily complex
        # because the cdist seems not to have a standardized Euclidean options
        # that takes a different V for every row. So I have to do the more
        # complicated Mahalanobis distance which requires a full covariance
        # matrix even though we have no co-variance terms -- only the diagonal

        # FIT vpd for this channel
        est_vpd, est_slope, est_xy = vpd_fitter(channel_uX, ch_dt_ann, n_dyes)
        check.array_t(est_vpd, ndim=1)
        channel_est_spd_count[ch] = est_xy
        channel_est_spd_slope[ch] = est_slope

        if test_data_dye_tracks is not None:
            # Truth is provided
            n_test_rows, n_test_channels, n_test_cycles = test_data_dye_tracks.shape
            assert n_test_channels == n_channels and n_test_cycles == n_cycles
            check.array_t(test_data_y, (n_test_rows,), int)
            true_dt_iz = ch_dt_ann.get_iz(test_data_dye_tracks[:, ch, :].astype(int))

            true_vpd, true_slope, true_xy = vpd_fitter(
                channel_uX, ch_dt_ann, n_dyes, cheat_dt_iz=true_dt_iz
            )
            channel_true_spd_count[ch] = true_xy
            channel_true_spd_slope[ch] = true_slope
'''
