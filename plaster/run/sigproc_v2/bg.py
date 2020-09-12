"""
Functions for measuring and removing background.
One entrypoint: bg_remove
"""

import cv2
import numpy as np
from plaster.tools.image import imops


def background_extract(im, kernel):
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
    fg_mask = np.where(cim > thresh, 1, 0)

    fg_mask = cv2.dilate(fg_mask.astype(np.uint8), circle, iterations=1)
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
    reg_bg, _ = background_regional_estimate_im(im, kernel)
    return bg_remove(im, reg_bg)



'''
def _background_stats_ims(flzl_ims, divs):
    """
    Loops over ims calling _background_regional_estimate_im

    Arguments:
        flzl_ims: frame, zslices ims to be analyzed (one channel)
        divs: divisions (in two dims) of image for regional stats

    Returns:
        bg_mean, bg_std averaged over all fields
    """
    check.array_t(flzl_ims, ndim=4)
    n_fields, n_zslices = flzl_ims.shape[0:2]

    fl_reg_bg_mean = np.zeros((n_fields, divs, divs))
    fl_reg_bg_std = np.zeros((n_fields, divs, divs))

    for fl_i in range(n_fields):
        for z_i in range(n_zslices):
            reg_bg_mean, reg_bg_std = _background_regional_estimate_im(
                flzl_ims[fl_i, z_i], divs
            )
            fl_reg_bg_mean[fl_i, :, :] = reg_bg_mean
            fl_reg_bg_std[fl_i, :, :] = reg_bg_std

    return np.mean(fl_reg_bg_mean, axis=0), np.mean(fl_reg_bg_std, axis=0)
'''
