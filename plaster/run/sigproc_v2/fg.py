import numpy as np
import time
from plaster.run.sigproc_v2 import bg
from plaster.run.calib.calib import RegPSF, approximate_psf
from plaster.run.sigproc_v2.c_gauss2_fitter import gauss2_fitter
from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import (
    AugmentedGauss2Params,
    Gauss2Params,
)
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, ROI, WH, XY, YX
from plaster.tools.schema import check
from plaster.tools.utils.data import one_sided_nanstd

from plaster.run.sigproc_v2.c_radiometry.radiometry import radiometry_field_stack
from plaster.tools.log.log import debug, important, prof


def peak_find(im, approx_psf):
    """
    Peak find on a single image.

    In some cases this im might be a mean of multiple channels
    in other cases it might stand-alone on a single channel.

    Arguments:
        im: the image to peak find
        approx_psf: An estimated PSF search kernel
        bg_std:
            The standard deviation of the background,
            this is scaled by 1.25 to pick a threshold

    Returns:
        locs: ndarray (n_peaks_found, 2) where the 2 is in (y,x) order
    """
    from skimage.feature import peak_local_max  # Defer slow import

    std = one_sided_nanstd(im.flatten())
    thresh = 2 * std  # 2 std found empirically

    cim = imops.convolve(np.nan_to_num(im, nan=float(np.nanmedian(im))), approx_psf)

    # CLEAN the edges
    # ZBS: Added because there were often edge effect from the convolution
    # that created false stray edge peaks.
    imops.edge_fill(cim, approx_psf.shape[0])

    # The background is well-described by the the histogram centered
    # around zero thanks to the fact that im and kern are expected
    # to be roughly zero-centered. Therefore we estimate the threshold
    # by using the samples less than zero cim[cim<0]
    if (cim < 0).sum() > 0:
        cim[cim < thresh] = 0
        return peak_local_max(cim, min_distance=2, threshold_abs=thresh)
    else:
        return np.zeros((0, 2))


def _sub_pixel_peak_find(im, peak_dim, locs):
    """
    This is a subtle calculation.

    locs is given as an *integer* position (only has pixel accuracy).
    We then extract out a sub-image using an *integer* half width.
    Peak_dim is typically odd. Suppose it is (11, 11)
    That makes half_peak_mea_i be 11 // 2 = 5

    Suppose that a peak is at (17.5, 17.5).

    Suppose that peak was found a (integer) location (17, 17)
    which is within 1 pixel of its center as expected.

    We extract the sub-image at (17 - 5, 17 - 5) = (12:23, 12:23)

    The Center-of-mass calculation should return (5.5, 5.5) because that is
    relative to the sub-image which was extracted

    We wish to return (17.5, 17.5). So that's the lower left
    (17 - 5) of the peak plus the COM found.
    """
    check.array_t(locs, dtype=int)
    assert peak_dim[0] == peak_dim[1]
    half_peak_mea_i = peak_dim[0] // 2
    lower_left_locs = locs - half_peak_mea_i
    com_per_loc = np.zeros(locs.shape)
    for loc_i, loc in enumerate(lower_left_locs):
        peak_im = imops.crop(im, off=YX(loc), dim=peak_dim, center=False)
        com_per_loc[loc_i] = imops.com(peak_im ** 2)
    return lower_left_locs + com_per_loc


def sub_pixel_peak_find(im, approx_psf):
    """
    First find peaks with pixel accuracy and then go back over each
    one and use the center of mass method to sub-locate them
    """
    locs = peak_find(im, approx_psf).astype(int)
    return _sub_pixel_peak_find(im, HW(approx_psf.shape), locs)


def radiometry_one_channel_one_cycle_fit_method(im, reg_psf: RegPSF, locs):
    """
    Like radiometry_one_channel_one_cycle() but using a gaussian fit

    Returns:
        11 typle:
            signal, noise, aspect_ratio, fit parameters
    """
    check.array_t(locs, ndim=2, shape=(None, 2))
    ret_params, _ = fit_image_with_reg_psf(im, locs, reg_psf)
    return ret_params


"""
This is the beginning of a zap but the hangup is that
I'm accumlating to fg and cnt and would need to change
that to write out and then do the accum

def _do_?(im, approx_psf, reg_psf, ):
    im_no_bg, bg_std = bg.bg_estimate_and_remove(fl_ims[fl_i], approx_psf)

    # FIND PEAKS
    locs = peak_find(im_no_bg, approx_psf)

    # RADIOMETRY
    # signals, _, _ = radiometry_one_channel_one_cycle(im_no_bg, reg_psf, locs)
    radmat = radiometry_field_stack(
        im_no_bg[None, None, :, :],
        locs=locs.astype(float),
        reg_psf=reg_psf,
        focus_adjustment=np.ones((1,), dtype=float),
    )
    # radmat is: (n_peaks, n_channels, n_cycles, 4)
    assert radmat.shape[1] == 1  # TODO: Multichannel
    signals = radmat[:, 0, 0, 0]

    # FIND outliers
    if not np.all(np.isnan(signals)):
        low, high = np.nanpercentile(signals, (10, 90))

        # SPLAT circles of the intensity of the signal into an accumulator
        for loc, sig in zip(locs, signals):
            if low <= sig <= high:
                # TODO: The CENTER=FALSE here smells wrong
                imops.accum_inplace(fg, sig * circle, loc, center=False)
                imops.accum_inplace(cnt, circle, loc, center=False)
"""


def fg_estimate(fl_ims, reg_psf: RegPSF, bandpass_kwargs):
    """
    Estimate the foreground illumination averaged over every field for
    one channel on the first cycle.

    This has a chicken-and-egg quality to it because we need to extract
    the foreground in order to estimate it!

    To resolve the chicken-and-egg we run the minimal peak finder on only
    the first cycle.

    Thus we:
        Remove the background
        Find the peaks
        Compute radiometry
        Keep the reasonable radiometry (ie percentile 10-90)
        Splat the radiometry as circles into a new image
        Make a regional summary
    """

    approx_psf = approximate_psf()
    n_fields = fl_ims.shape[0]
    dim = fl_ims.shape[-2:]

    # SANITY CHECK that z_reg_psfs
    # assert psf.psf_validate(z_reg_psfs)

    # ALLOCATE two accumulators: one for the signals and one for the counts
    fg = np.zeros(dim)
    cnt = np.zeros(dim)

    # ALLOCATE a circle mask to use to splat in the values
    circle = imops.generate_circle_mask(7).astype(float)

    # TODO: Replace with a zap over fields (see notes in commented out above)
    for fl_i in range(n_fields):
        filtered_im, bg_std = bg.bandpass_filter(fl_ims[fl_i], **bandpass_kwargs,)
        locs = peak_find(filtered_im, approx_psf)

        # RADIOMETRY
        # signals, _, _ = radiometry_one_channel_one_cycle(im_no_bg, reg_psf, locs)
        im = np.ascontiguousarray(filtered_im[None, None, :, :])
        radmat = radiometry_field_stack(
            im,
            locs=locs.astype(float),
            reg_psf=reg_psf,
            focus_adjustment=np.ones((1,), dtype=float),
        )
        # radmat is: (n_peaks, n_channels, n_cycles, 4)
        assert radmat.shape[1] == 1  # TODO: Multichannel
        signals = radmat[:, 0, 0, 0]

        # FIND outliers
        if not np.all(np.isnan(signals)):
            low, high = np.nanpercentile(signals, (10, 90))

            # SPLAT circles of the intensity of the signal into an accumulator
            for loc, sig in zip(locs, signals):
                if low <= sig <= high:
                    # TODO: The CENTER=FALSE here smells wrong
                    imops.accum_inplace(fg, sig * circle, loc, center=False)
                    imops.accum_inplace(cnt, circle, loc, center=False)

    # Fill nan into all places that had no counts
    fg[cnt == 0] = np.nan

    # Average over the samples (fg / cnt)
    mean_im = fg / cnt

    def median_nan_ok(arr):
        if np.all(np.isnan(arr)):
            return np.nan
        else:
            return np.nanmedian(arr)

    # BALANCE regionally. 10 is an empirically found size
    bal = imops.region_map(mean_im, median_nan_ok, 10)

    # RETURN the balance adjustment. That is, multiply by this matrix
    # to balance an image. In other words, the brightest region will == 1.0
    return np.nanmax(bal) / bal, mean_im


def fit_image_with_reg_psf(im, locs, reg_psf: RegPSF):
    assert isinstance(reg_psf, RegPSF)

    reg_yx = np.clip(
        np.floor(reg_psf.n_divs * locs / im.shape[0]).astype(int),
        a_min=0,
        a_max=reg_psf.n_divs - 1,
    )

    n_locs = len(locs)
    guess_params = np.zeros((n_locs, AugmentedGauss2Params.N_FULL_PARAMS))

    # COPY over parameters by region for each peak
    guess_params[:, Gauss2Params.SIGMA_X] = reg_psf.get_params(
        reg_yx[:, 0], reg_yx[:, 1], RegPSF.SIGMA_X,
    )
    guess_params[:, Gauss2Params.SIGMA_Y] = reg_psf.get_params(
        reg_yx[:, 0], reg_yx[:, 1], RegPSF.SIGMA_Y,
    )
    guess_params[:, Gauss2Params.RHO] = reg_psf.get_params(
        reg_yx[:, 0], reg_yx[:, 1], RegPSF.RHO,
    )

    # CENTER
    guess_params[:, Gauss2Params.CENTER_X] = reg_psf.peak_mea / 2
    guess_params[:, Gauss2Params.CENTER_Y] = reg_psf.peak_mea / 2

    # Pass zero to amp and offset to force the fitter to make its own guess
    guess_params[:, Gauss2Params.AMP] = 0.0
    guess_params[:, Gauss2Params.OFFSET] = 0.0

    return gauss2_fitter.fit_image(im, locs, guess_params, reg_psf.peak_mea)


def focus_from_fitmat(fitmat, reg_psf: RegPSF):
    """
    fitmat: (n_peaks, n_channels, n_cycles, AugmentedGauss2Params.N_FULL_PARAMS)
    """
    focus_const = 1.0
    n_peaks, n_channels, n_cycles, n_params = fitmat.shape
    assert n_channels == 1  # TODO: Multichannel
    focus_per_cycle = []
    for cy_i in range(n_cycles):
        ch_fitmat = fitmat[:, 0, cy_i, :]

        fit_sig_x = ch_fitmat[:, Gauss2Params.SIGMA_X]
        fit_sig_y = ch_fitmat[:, Gauss2Params.SIGMA_Y]

        keep_mask = (
            (1.0 < fit_sig_x)
            & (fit_sig_x < 1.5)
            & (1.0 < fit_sig_y)
            & (fit_sig_y < 1.5)
        )

        fit_sigma = np.nanmean(
            np.concatenate((fit_sig_x[keep_mask], fit_sig_y[keep_mask]))
        )
        psf_sigma = np.mean(
            np.concatenate(
                (
                    reg_psf.params[:, :, RegPSF.SIGMA_X],
                    reg_psf.params[:, :, RegPSF.SIGMA_Y],
                )
            )
        )

        focus_per_cycle += [focus_const * fit_sigma / psf_sigma]
    return np.array(focus_per_cycle)


def cycle_balance_one_channel(sig, one_count_mean, one_count_std):
    """
    Compute a cycle balance based on the 1-count radiomerty

    Usage
        sig = run.sigproc_v2.sig()
        correction_per_cycle = cycle_balance(sig, 5000.0, 1000.0)
        corr_sig = sig * correction_per_cycle
    """
    check.array_t(sig, ndim=2)  # (n_peaks, n_cycles)
    low = one_count_mean - one_count_std
    high = one_count_mean + one_count_std
    _sig = np.where((low < sig) & (sig < high), sig, np.nan)
    per_cycle = np.nanmedian(_sig, axis=0)
    correct_rad = np.mean(per_cycle)
    correction_per_cycle = correct_rad / per_cycle
    return correction_per_cycle
