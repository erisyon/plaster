"""
Functions for measuring and removing background.
One entrypoint: bg_remove
"""

import cv2
import numpy as np
from plaster.tools.image import imops
from plaster.tools.schema import check
from plaster.tools.utils import data
from plaster.tools.log.log import debug


def bandpass_filter(im, low_inflection, low_sharpness, high_inflection, high_sharpness):
    """
    Use a band-pass filter to subtract out background and "bloom" which is
    the light that scatters from foreground to background.

    Note: A low_inflection of -10 effectively removes the low-pass filter
    and a high_inflection of +10 effectively removes the high-pass filter

    Values of sharpness = 50.0 are usually fine.

    Returns the filtered image

    """

    # These number were hand-tuned to Abbe (512x512) and might be wrong for other
    # sizes/instruments and will need to be derived and/or calibrated.
    check.array_t(im, ndim=2, is_square=True, dtype=np.float64)
    low_cut = imops.generate_center_weighted_tanh(
        im.shape[0], inflection=low_inflection, sharpness=low_sharpness
    )
    high_cut = 1 - imops.generate_center_weighted_tanh(
        im.shape[0], inflection=high_inflection, sharpness=high_sharpness
    )
    filtered_im = imops.fft_filter_with_mask(im, mask=low_cut * high_cut)

    # The filters do not necessarily create a zero-centered background so
    # not remove the median to pull the background to zero.
    filtered_im -= np.median(filtered_im)

    # The bg_std is used later for tuning the peak finder.
    # Once I convert full to band-pass filter then this can just be eliminated
    # because I think it will be a constant. For now, I'm keeping
    # backward compatibility with bg_estimate_and_remove and setting
    # the constant here.
    bg_std = 3.0 * data.one_sided_nanstd(
        filtered_im.flatten(), mean=0.0, negative_side=True
    )

    return filtered_im, bg_std


def background_extract(im, kernel, dilate=1):
    """
    Using an approximate peak kernel, separate FG and BG regionally
    and return the bg mean and std.

    Arguments:
        im: a single frame

    Returns:
        bg_im, fg_mask
        bg_im has nan's where the fg_mask is True
    """
    # mask_radius in pixels of extra space added around FG candidates
    mask_radius = 2  # Empirical

    circle = imops.generate_circle_mask(mask_radius).astype(np.uint8)

    # Note: imops.convolve require float64 inputs; im is likely to be float32,
    #      so we have to cast it to float64.  Alternatively we could investigate
    #      if imops.convolve really ought to require float64?
    med = float(np.nanmedian(im))
    cim = imops.convolve(np.nan_to_num(im.astype(np.float64), nan=med), kernel)

    # cim can end up with artifacts around the nans to the nan_mask
    # is dilated and splated as zeros back over the im
    nan_mask = cv2.dilate(np.isnan(im).astype(np.uint8), circle, iterations=1)

    # The negative side of the convoluted image has no signal
    # so the std of the symmetric distribution (reflecting the
    # negative side around zero) is a good estimator of noise.
    if (cim < 0).sum() == 0:
        # Handle the empty case to avoid warning
        thresh = 1e10
    else:
        thresh = np.nanstd(np.concatenate((cim[cim < 0], -cim[cim < 0])))
        thresh = np.nan_to_num(
            thresh, nan=1e10
        )  # For nan thresh just make them very large
    cim = np.nan_to_num(cim)
    fg_mask = np.where(cim > thresh, 1, 0).astype(bool)

    if dilate > 0:
        fg_mask = cv2.dilate(fg_mask.astype(np.uint8), circle, iterations=dilate)
    bg_im = np.where(fg_mask | nan_mask, np.nan, im)
    return bg_im, fg_mask


def background_regional_estimate_im_with_bg_im(bg_im, divs=64, inpaint=True):
    """
    See _background_regional_estimate_im

    bg_im: Should have nan's where the FG pixels are
    """

    def nanstats(dat):
        if np.all(np.isnan(dat)):
            return np.nan, np.nan
        return np.nanmedian(dat), np.nanstd(dat)

    reg_bg_mean, reg_bg_std = imops.region_map(bg_im, nanstats, divs=divs)

    if inpaint:
        reg_bg_mean = cv2.inpaint(
            np.nan_to_num(reg_bg_mean.astype(np.float32)),
            np.isnan(reg_bg_mean).astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )
        reg_bg_std = cv2.inpaint(
            np.nan_to_num(reg_bg_std.astype(np.float32)),
            np.isnan(reg_bg_std).astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )

    assert not np.any(np.isnan(reg_bg_mean))
    assert not np.any(np.isnan(reg_bg_std))
    return reg_bg_mean, reg_bg_std


def background_regional_estimate_im(im, kernel, divs=64, inpaint=True):
    """
    Using an approximate peak kernel, separate FG and BG regionally
    and return the bg mean and std.

    Arguments:
        im: a single frame
        divs:
            Regional divisions (both horiz and vert)
        inpaint: If True then fill NaNs

    Returns:
        regional bg_mean and bg_std
    """

    bg_im, _ = background_extract(im, kernel)
    return background_regional_estimate_im_with_bg_im(bg_im, divs=divs, inpaint=inpaint)


def bg_remove(im, reg_bg):
    """
    Expand the reg_bg to match im and remove it.
    """
    bg_im = imops.interp(reg_bg, im.shape[-2:])
    return im - bg_im


def bg_estimate_and_remove(im, kernel):
    """
    Extract the bg and subtract it
    """
    reg_bg, reg_std = background_regional_estimate_im(im, kernel)
    return bg_remove(im, reg_bg), np.nanmean(reg_std)
