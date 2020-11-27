import numpy as np

from plaster.run.sigproc_v2 import bg
from plaster.run.sigproc_v2.reg_psf import RegPSF, approximate_psf
from plaster.run.sigproc_v2.c_gauss2_fitter import gauss2_fitter
from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import (
    AugmentedGauss2Params,
    Gauss2Params,
)
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, ROI, WH, XY, YX
from plaster.tools.schema import check
from plaster.tools.log.log import debug, important, prof


def peak_find(im, approx_psf, bg_std):
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

    thresh = 1.25 * bg_std  # This 1.25 was found empirically

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
    check.array_t(locs, dtype=int)
    assert peak_dim[0] == peak_dim[1]
    half_peak_mea = peak_dim[0] // 2
    com_per_loc = np.zeros(locs.shape)
    for loc_i, loc in enumerate(locs):
        peak_im = imops.crop(im, off=YX(loc[0], loc[1]), dim=peak_dim, center=True)
        com_per_loc[loc_i] = imops.com(peak_im ** 2) - half_peak_mea
    return locs + com_per_loc


def sub_pixel_peak_find(im, approx_psf, bg_std):
    """
    First find peaks with pixel accuracy and then go back over each
    one and use the center of mass method to sub-locate them
    """
    locs = peak_find(im, approx_psf, bg_std).astype(int)
    return _sub_pixel_peak_find(im, HW(approx_psf.shape), locs)


def _radiometry_one_peak(
    peak_im, psf_kernel, allow_non_unity_psf_kernel=False, allow_subpixel_shift=True,
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
    assert peak_im.shape == psf_kernel.shape

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

    if allow_subpixel_shift:
        com_before = imops.com(peak_im ** 2)
        center_pixel = np.array(peak_im.shape) / 2
        peak_im = imops.sub_pixel_shift(peak_im, center_pixel - com_before)

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

    # COMPUTE aspect ratio
    aspect_ratio = imops.distribution_aspect_ratio(peak_im)

    def distribution_eigen(im):
        ys, xs = np.indices(im.shape)
        pos = np.array((ys, xs)).T.reshape(-1, 2).astype(float)
        mas = im.T.reshape(-1)
        com_y = (pos[:, 0] * mas).sum() / im.sum()
        com_x = (pos[:, 1] * mas).sum() / im.sum()
        com = np.array([com_y, com_x])
        centered = pos - com
        dy = centered[:, 0] * mas
        dx = centered[:, 1] * mas
        cov = np.cov(np.array([dy, dx]))
        eig_vals, eig_vecs = LA.eig(cov)
        return eig_vals, eig_vecs, cov

    return signal, noise, aspect_ratio


def radiometry_one_channel_one_cycle(im, reg_psf: RegPSF, locs):
    """
    TODO: Convert this to C

    Use the PSFs to compute the Area-Under-Curve of the data in chcy_ims
    for each peak location of locs.

    Arguments:
        im: One image (one channel, cycle)
        reg_psf: (n_z_slices, divs, divs, peak_mea, peak_mea)
        locs: (n_peaks, 2). The second dimension is in (y, x) order

    Returns:
        signal, noise, aspect_ratio
    """
    check.array_t(im, ndim=2)
    check.t(reg_psf, RegPSF)
    check.array_t(locs, ndim=2, shape=(None, 2))

    n_locs = len(locs)
    div_locs = imops.locs_to_region(locs, reg_psf.n_divs, im.shape)

    signal = np.full((n_locs,), np.nan)
    noise = np.full((n_locs,), np.nan)
    aspect_ratio = np.full((n_locs,), np.nan)

    psf_ims = reg_psf.render()
    psf_dim = HW(reg_psf.peak_mea, reg_psf.peak_mea)

    for loc_i, (loc, div_loc) in enumerate(zip(locs, div_locs)):
        peak_im = imops.crop(im, off=YX(loc), dim=psf_dim, center=True)
        if peak_im.shape != psf_dim:
            # Skip near edges
            continue

        if np.any(np.isnan(peak_im)):
            # Skip nan collisions
            continue

        psf_kernel = psf_ims[div_loc[0], div_loc[1]]
        _signal, _noise, _aspect_ratio = _radiometry_one_peak(peak_im, psf_kernel)

        signal[loc_i] = _signal
        noise[loc_i] = _noise
        aspect_ratio[loc_i] = _aspect_ratio

    return signal, noise, aspect_ratio


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


def fg_estimate(fl_ims, reg_psf: RegPSF, progress=None):
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

    for fl_i in range(n_fields):
        # REMOVE BG
        if progress:
            progress(fl_i, n_fields, False)

        im_no_bg, bg_std = bg.bg_estimate_and_remove(fl_ims[fl_i], approx_psf)

        # FIND PEAKS
        locs = peak_find(im_no_bg, approx_psf, bg_std)

        # RADIOMETRY
        signals, _, _ = radiometry_one_channel_one_cycle(im_no_bg, reg_psf, locs)

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
    guess_params[:, 0 : Gauss2Params.N_PARAMS] = reg_psf.params[
        reg_yx[:, 0], reg_yx[:, 1], 0 : Gauss2Params.N_PARAMS,
    ]

    # CENTER
    guess_params[:, Gauss2Params.CENTER_X] = reg_psf.peak_mea / 2
    guess_params[:, Gauss2Params.CENTER_Y] = reg_psf.peak_mea / 2

    # Pass zero to amp and offset to force the fitter to make its own guess
    guess_params[:, Gauss2Params.AMP] = 0.0
    guess_params[:, Gauss2Params.OFFSET] = 0.0

    return gauss2_fitter.fit_image(im, locs, guess_params, reg_psf.peak_mea)
