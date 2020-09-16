import numpy as np
from plaster.run.sigproc_v2 import bg
from plaster.run.sigproc_v2 import psf
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, ROI, WH, XY, YX
from plaster.tools.log.log import debug, important
from plaster.tools.schema import check


def peak_find(im, kernel):
    """
    Peak find on a single image.

    In some cases this im might be a mean of multiple channels
    in other cases it might stand-alone on a single channel.

    Returns:
        locs: ndarray (n_peaks_found, 2) where the 2 is in (y,x) order
    """
    from skimage.feature import peak_local_max  # Defer slow import

    cim = imops.convolve(np.nan_to_num(im, nan=float(np.nanmedian(im))), kernel)

    # The background is well-described by the the histogram centered
    # around zero thanks to the fact that im and kern are expected
    # to be roughly zero-centered. Therefore we estimate the threshold
    # by using the samples less than zero cim[cim<0] and taking the 99th percentile
    if (cim < 0).sum() > 0:
        thresh = np.percentile(-cim[cim < 0], 99)
        cim[cim < thresh] = 0
        return peak_local_max(cim, min_distance=2, threshold_abs=thresh)
    else:
        return np.zeros((0, 2))


def _fit_focus(z_reg_psfs, locs, im):
    """
    Each image may have a slightly different focus due to drift of the z-axis on
    the instrument.

    During calibration we generated a regional-PSF as a function of z.
    This is called the "z_reg_psf" and has shape like:
    (13, 5, 5, 11, 11) where:
        (13) is the 13 z-slices where slice 6 is the most-in-focus.
        (5, 5) is the regionals divs
        (11, 11) are the pixels of the PSF peaks

    Here we sub-sample peaks locs on im to decide which
    PSF z-slice best describes this images.

    Note, if the instrument was perfect at maintaining the z-focus
    then this function would ALWAYS return 6.
    """

    # TODO: randomly sample a sub-set of locs and pick the correct
    # regional PSF and fit every z-stack to the sample.
    # For each randomly sanpled loc we will have a best
    # z-index. Then we take the plurality vote of that.

    # Until then:
    return z_reg_psfs.shape[0] // 2


def _radiometry_one_peak(
    peak_im, psf_kernel, center_weighted_mask, allow_non_unity_psf_kernel=False
):
    """
    Helper for _analyze_step_6_radiometry() to compute
    radiometry on a single peak.

    Arguments:
        peak_im: a small regional image of a peak roughly centered.
                 This expected to be from a regionally balance and channel equalized
                 source image with the regional background already subtracted

        psf_kernel: The kernel appropriate for the region (from calibration)

    Returns:
        signal: The area under the kernel (always >= 0)
        noise: The standard deviation of the residuals (always >= 0)

    Notes:
        I think that allow_non_unity_psf_kernel is no longer needed. It only
        seems to be referenced in a (probably old) notebook.
    """
    check.array_t(peak_im, ndim=2, is_square=True)
    check.array_t(psf_kernel, ndim=2, is_square=True)
    check.array_t(center_weighted_mask, ndim=2, is_square=True)
    assert peak_im.shape == psf_kernel.shape
    assert psf_kernel.shape == center_weighted_mask.shape

    if not allow_non_unity_psf_kernel:
        try:
            assert 1.0 - np.sum(psf_kernel) < 1e-6
        except AssertionError:
            debug("np.sum(psf_kernel)", np.sum(psf_kernel))
            raise

    # Weight the peak_im by the centering_kernel to eliminate
    # noise from neighbors during COM calculations

    # SHIFT peak_im to center with sub-pixel alignment
    # Note, we scale peak_im by the centering_kernel so that
    # the COM will not be polluted by neighbors

    com_before = imops.com((center_weighted_mask * peak_im) ** 2)
    center_pixel = np.array(peak_im.shape) / 2
    peak_im = center_weighted_mask * imops.sub_pixel_shift(
        peak_im, center_pixel - com_before
    )

    # WEIGH the data with the psf_kernel and then normalize
    # by the psf_kernel_sum_squared to estimate signal
    psf_kernel_sum_squared = np.sum(psf_kernel ** 2)
    signal = 0.0
    if psf_kernel_sum_squared > 0.0:
        signal = np.sum(psf_kernel * peak_im) / psf_kernel_sum_squared

    # COMPUTE the noise by examining the residuals
    residuals = peak_im - signal * psf_kernel
    var_residuals = np.var(residuals)
    noise = 0.0
    if psf_kernel_sum_squared > 0.0:
        noise = np.sqrt(var_residuals / psf_kernel_sum_squared)

    return signal, noise


def radiometry_one_channel_one_cycle(im, z_reg_psfs, locs):
    """
    Use the PSFs to compute the Area-Under-Curve of the data in chcy_ims
    for each peak location of locs.

    Arguments:
        im: One image (one channel, cycle)
        z_reg_psfs: (n_z_slices, divs, divs, peak_mea, peak_mea)
        locs: (n_peaks, 2). The second dimension is in (y, x) order

    Returns:
        signal, noise
    """
    check.array_t(im, ndim=2)
    check.array_t(z_reg_psfs, ndim=5)
    check.array_t(locs, ndim=2, shape=(None, 2))

    n_z_slices, divs, _, peak_mea, _ = z_reg_psfs.shape
    n_locs = len(locs)

    signal = np.full((n_locs,), np.nan)
    noise = np.full((n_locs,), np.nan)

    psf_dim = z_reg_psfs.shape[-2:]
    assert z_reg_psfs.shape[1] == divs
    assert z_reg_psfs.shape[2] == divs
    assert z_reg_psfs.shape[3] == peak_mea
    assert z_reg_psfs.shape[4] == peak_mea

    # TODO: Try removing this center_weighted_mask, I suspect it makes things worse
    # center_weighted_mask = imops.generate_center_weighted_tanh(
    #    peak_mea, radius=2.0
    # )
    # All ones center_weighted_mask
    center_weighted_mask = np.ones(psf_dim)

    # TASK: Eventually this will examine which z-depth of the PSFs is best fit for this cycle.
    # The result will be a per-cycle index into the chcy_regional_psfs
    # Until then the index is hard-coded to the middle index of regional_psf_zstack
    # See _fit_focus
    best_focus_zslice_i = _fit_focus(z_reg_psfs, locs, im)

    reg_psfs = z_reg_psfs[best_focus_zslice_i, :, :, :, :]
    for loc_i, loc in enumerate(locs):
        peak_im = imops.crop(im, off=YX(loc), dim=HW(psf_dim), center=True)
        if peak_im.shape != psf_dim:
            # Skip near edges
            continue

        if np.any(np.isnan(peak_im)):
            # Skip nan collisions
            continue

        psf_kernel = reg_psfs[
            int(divs * loc[0] / im.shape[0]), int(divs * loc[1] / im.shape[1]),
        ]

        if np.sum(psf_kernel) == 0.0:
            _signal, _noise = np.nan, np.nan
        else:
            _signal, _noise = _radiometry_one_peak(
                peak_im, psf_kernel, center_weighted_mask=center_weighted_mask
            )

        signal[loc_i] = _signal
        noise[loc_i] = _noise

    return signal, noise


def fg_estimate(fl_ims, z_reg_psfs):
    """
    Estimate the foreground illumination averaged over every field for
    one channel on the first cycle.

    This has a chicken-and-egg quality to it because we need to extract
    the foreground in order to estimate it!

    To resolve the chicken-and-egg we run the minimal peak finder on only
    the first cycle.

    Thus we:
        Remove bg the BG
        Find the peaks
        Compute radiometry
        Keep the reasonable radiometry (ie percentile 10-90)
        Splat the radiometry as circles into a new image
        Make a regional summary
    """

    kernel = psf.approximate_kernel()
    n_fields = fl_ims.shape[0]
    dim = fl_ims.shape[-2:]

    # ALLOCATE two accumulators: one for the signals and one for the counts
    fg = np.zeros(dim)
    cnt = np.zeros(dim)

    # ALLOCATE a circle mask to use to splat in the values
    circle = imops.generate_circle_mask(7).astype(float)

    for fl_i in range(n_fields):
        # REMOVE BG
        im_no_bg = bg.bg_estimate_and_remove(fl_ims[fl_i], kernel)

        # FIND PEAKS
        locs = peak_find(im_no_bg, kernel)

        # RADIOMETRY
        signals, _ = radiometry_one_channel_one_cycle(im_no_bg, z_reg_psfs, locs)

        # FIND outliers
        low, high = np.nanpercentile(signals, (10, 90))

        # SPLAT circles of the intensity of the signal into an accumulator
        for loc, sig in zip(locs, signals):
            if low <= sig <= high:
                imops.accum_inplace(fg, sig * circle, loc, center=False)
                imops.accum_inplace(cnt, circle, loc, center=False)

    # Fill nan into all places that had no counts
    fg[cnt == 0] = np.nan

    # Average over the samples (fg / cnt)
    mean_im = fg / cnt

    # BALANCE regionally. 10 is an empirally found size
    bal = imops.region_map(mean_im, np.nanmedian, 10)

    # RETURN the balance adjustment. That is, multiply by this matrix
    # to balance an image. In other words, the brightest region will == 1.0
    return np.max(bal) / bal, mean_im