from enum import IntEnum
import math
import cv2
import numpy as np
from plaster.run.sigproc_v2 import bg
from plaster.run.sigproc_v2 import fg
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, ROI, WH, XY, YX
from plaster.tools.image.imops import sub_pixel_center
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.zap import zap
from plaster.tools.log.log import debug, important


def approximate_kernel():
    """
    Return a zero-centered AUC=1.0 2D Gaussian for peak finding
    """
    std = 1.5  # This needs to be tuned and may be instrument dependent
    mea = 11
    kern = imops.gauss2_rho_form(
        amp=1.0,
        std_x=std,
        std_y=std,
        pos_x=mea // 2,
        pos_y=mea // 2,
        rho=0.0,
        const=0.0,
        mea=mea,
    )
    return kern - np.mean(kern)


class PSFEstimateMaskFields(IntEnum):
    """Mask fields returned as the second return of psf_estimate"""

    considered = 0
    skipped_near_edges = 1
    skipped_too_crowded = 2
    skipped_has_nan = 3
    skipped_empty = 4
    skipped_too_dark = 5
    skipped_too_oval = 6
    accepted = 7


def _psf_accumulate(
    im, locs, mea, keep_dist=8, threshold_abs=None, return_reasons=True
):
    """
    Given a single im, typically a regional sub-image, accumulate
    PSF evidence from each locs that meets a set of criteria.

    Any one image may not produce enough (or any) candidate spots and it
    is therefore expected that this function is called over a large number
    of fields to get sufficient samples.

    Arguments:
        im: Expected to be a single field, channel, cycle (BG already removed).
        locs: array (n, 2) in coordinates of im. Expected to be well-separated
        mea: The peak_measure (must be odd)
        threshold_abs: The average pixel brightness to accept the peak
        keep_dist: Pixels distance to determine crowding

    Returns:
        psf: ndarray (mea, mea) image
        reason_counts: An array of masks of why peaks were accepted/rejected
            See PSFEstimateMaskFields for the columns
    """
    from scipy.spatial.distance import cdist  # Defer slow import

    n_locs = len(locs)
    dist = cdist(locs, locs, metric="euclidean")
    dist[dist == 0.0] = np.nan

    if not np.all(np.isnan(dist)):
        closest_dist = np.nanmin(dist, axis=1)
    else:
        closest_dist = np.zeros(n_locs)

    # Aligned peaks will accumulate into this psf matrix
    dim = (mea, mea)
    dim2 = (mea + 2, mea + 2)
    psf = np.zeros(dim)

    n_reason_mask_fields = len(PSFEstimateMaskFields)
    reason_masks = np.zeros((n_locs, n_reason_mask_fields))

    for i, (loc, closest_neighbor_dist) in enumerate(zip(locs, closest_dist)):
        reason_masks[i, PSFEstimateMaskFields.considered] = 1

        # EXTRACT a peak with extra pixels around the edges (dim2 not dim)
        peak_im = imops.crop(im, off=YX(loc), dim=HW(dim2), center=True)

        if peak_im.shape != dim2:
            # Skip near edges
            reason_masks[i, PSFEstimateMaskFields.skipped_near_edges] = 1
            continue

        if closest_neighbor_dist < keep_dist:
            reason_masks[i, PSFEstimateMaskFields.skipped_too_crowded] = 1
            continue

        if np.any(np.isnan(peak_im)):
            reason_masks[i, PSFEstimateMaskFields.skipped_has_nan] = 1
            continue

        # Sub-pixel align the peak to the center
        assert not np.any(np.isnan(peak_im))
        centered_peak_im = sub_pixel_center(peak_im)

        # Removing ckipping as the noise should cancel out
        # centered_peak_im = np.clip(centered_peak_im, a_min=0.0, a_max=None)
        peak_max = np.max(centered_peak_im)
        if peak_max == 0.0:
            reason_masks[i, PSFEstimateMaskFields.skipped_empty] = 1
            continue

        if threshold_abs is not None and peak_max < threshold_abs:
            # Reject spots that are not active
            reason_masks[i, PSFEstimateMaskFields.skipped_too_dark] = 1
            continue

        r = imops.distribution_aspect_ratio(centered_peak_im)
        if r > 2.0:
            reason_masks[i, PSFEstimateMaskFields.skipped_too_oval] = 1
            continue

        # TRIM off the extra now
        centered_peak_im = centered_peak_im[1:-1, 1:-1]

        psf += centered_peak_im / np.sum(centered_peak_im)
        reason_masks[i, PSFEstimateMaskFields.accepted] = 1

    n_accepted = np.sum(reason_masks[:, PSFEstimateMaskFields.accepted])
    if n_accepted > 0:
        psf /= np.sum(psf)
        assert np.min(psf) >= 0.0

    if return_reasons:
        return psf, reason_masks

    return psf


def _psf_one_z_slice(im, locs, divs=5, keep_dist=8, peak_mea=11):
    """
    Run PSF calibration for one image.

    These are typically combined from many fields and for each channel
    to get a complete calibration.

    This returns the accepted locs so that a z-stack can be estimated
    by using the most in-focus frame for the locations

    Arguments:
        im: One image, already background subtracted
        locs: The peak locations
        divs: Spatial divisions
        keep_dist: Pixel distance under which is considered a collision
        peak_mea: n pixel width and height to hold the peak image

    Returns:
        locs (location of accepted peaks)
        regional_psf_zstack
    """
    check.array_t(im, ndim=2)

    n_locs = locs.shape[0]
    accepted = np.zeros((n_locs,))

    # In each region gather a PSF estimate and a list of
    # locations that were accepted. These locs can be
    # re-used when analyzing other z slices
    reg_psfs = np.zeros((divs, divs, peak_mea, peak_mea))
    for win_im, y, x, coord in imops.region_enumerate(im, divs):
        mea = win_im.shape[0]
        assert win_im.shape[1] == mea

        local_locs = locs - coord
        local_locs_mask = np.all((local_locs > 0) & (local_locs < mea), axis=1)
        local_locs = local_locs[local_locs_mask]
        n_local_locs = local_locs.shape[0]

        psfs, reasons = _psf_accumulate(
            win_im, local_locs, peak_mea, keep_dist=keep_dist, return_reasons=True
        )
        reg_psfs[y, x] = psfs

        # DUMP reasons why the peaks were kept or rejected
        # for reason in (
        #     PSFEstimateMaskFields.accepted,
        #     # PSFEstimateMaskFields.skipped_near_edges,
        #     # PSFEstimateMaskFields.skipped_too_crowded,
        #     # PSFEstimateMaskFields.skipped_has_nan,
        #     # PSFEstimateMaskFields.skipped_empty,
        #     # PSFEstimateMaskFields.skipped_too_dark,
        #     # PSFEstimateMaskFields.skipped_too_oval,
        # ):
        #     n_local_rejected = (reasons[:, reason] > 0).sum()
        #     print(f"y,x={y},{x} {str(reason)}:, {n_local_rejected}")

        # Go backwards from local to global space.
        local_accepted_iz = np.argwhere(
            reasons[:, PSFEstimateMaskFields.accepted] == 1
        ).flatten()
        local_loc_i_to_global_loc_i = np.arange(n_locs)[local_locs_mask]
        assert local_loc_i_to_global_loc_i.shape == (n_local_locs,)

        global_accepted_iz = local_loc_i_to_global_loc_i[local_accepted_iz]
        accepted[global_accepted_iz] = 1

    return locs[accepted > 0], reg_psfs


def _do_psf_one_field_one_channel(zi_ims, peak_mea, divs, n_dst_zslices, n_src_zslices):
    """
    The worker for _psf_stats_one_channel()

    Arguments:
        zi_ims: stack of z slices of one field, one channel. It is not yet background subtracted.
        peak_mea: size of extracted square inside of which will be the PSF sample
        divs: regional divisions
        n_dst_zslices: NUmber of dst z_slices
        n_src_zslices: The range of z_slices centered on the most-in-focus to consider
    """
    assert n_dst_zslices % 2 == 1

    n_src_zslices_actual = zi_ims.shape[0]
    divs = divs
    peak_dim = (peak_mea, peak_mea)

    z_and_region_to_psf = np.zeros((n_dst_zslices, divs, divs, *peak_dim))

    dst_z_per_src_z = n_src_zslices / n_dst_zslices

    kernel = approximate_kernel()
    im_focuses = np.zeros((n_src_zslices_actual,))
    for src_zi in range(n_src_zslices_actual):
        im = zi_ims[src_zi]

        # ksize=9 was found empirically. The default is too small
        # and results in very bad focus estimation
        im_focuses[src_zi] = cv2.Laplacian(im, cv2.CV_64F, ksize=9).var()

    src_zi_best_focus = np.argmax(im_focuses)

    # FIND peaks on the best in focus and re-use those locs
    im = bg.bg_estimate_and_remove(zi_ims[src_zi_best_focus], kernel)
    locs = fg.peak_find(im, kernel)

    for dst_zi in range(n_dst_zslices):
        src_zi0 = math.floor(
            0.5
            + ((dst_zi - 0.5) - n_dst_zslices // 2) * dst_z_per_src_z
            + src_zi_best_focus
        )
        src_zi1 = math.floor(
            0.5
            + ((dst_zi + 0.5) - n_dst_zslices // 2) * dst_z_per_src_z
            + src_zi_best_focus
        )

        for src_zi in range(src_zi0, src_zi1):
            if 0 <= src_zi < n_src_zslices_actual:
                # Only if the source is inside the source range, accum to dst.
                im = zi_ims[src_zi]
                _, reg_psfs = _psf_one_z_slice(
                    im, divs=divs, peak_mea=peak_dim[0], locs=locs
                )
                z_and_region_to_psf[dst_zi] += reg_psfs

    return z_and_region_to_psf, im_focuses


def psf_normalize(z_and_region_to_psf):
    """
    The PSF tends to have some bias and needs to have a unit area-under-curve
    The bias is removed by fitting to a Gaussian including the offset
    and then removing the offset.
    """

    normalized = np.zeros_like(z_and_region_to_psf)

    n_z_slices, divs = z_and_region_to_psf.shape[0:2]
    for z_i in range(n_z_slices):
        for y in range(divs):
            for x in range(divs):

                psf = z_and_region_to_psf[z_i, y, x]

                if np.sum(psf) > 0:

                    # FIT to Gaussian to get the offset
                    fit_params, _ = imops.fit_gauss2(psf)
                    bias = fit_params[6]

                    psf = (psf - bias).clip(min=0)

                    # NORMALIZE so that all PSF estimates have unit area-under-curve
                    # The z_and_region_to_psf can have all-zero elements thus we use np_safe_divide below
                    denominator = psf.sum()
                    normalized[z_i, y, x] = utils.np_safe_divide(psf, denominator)

    return normalized


def psf_gaussianify(z_and_region_to_psf):
    """
    Fit to a Gaussian, remove bias, and resample
    """
    normalized = np.zeros_like(z_and_region_to_psf)
    h, w = z_and_region_to_psf.shape[-2:]
    n_z_slices, divs = z_and_region_to_psf.shape[0:2]
    for z_i in range(n_z_slices):
        for y in range(divs):
            for x in range(divs):

                psf = z_and_region_to_psf[z_i, y, x]

                if np.sum(psf) > 0:

                    # FIT to Gaussian to get the offset
                    fit_params, _ = imops.fit_gauss2(psf)
                    fit_params = list(fit_params)
                    fit_params[6] = 0
                    fit_params[3] = h // 2
                    fit_params[4] = w // 2
                    psf = imops.gauss2_rho_form(*fit_params)

                    # NORMALIZE so that all PSF estimates have unit area-under-curve
                    # The z_and_region_to_psf can have all-zero elements thus we use np_safe_divide below
                    denominator = psf.sum()
                    normalized[z_i, y, x] = utils.np_safe_divide(psf, denominator)

    return normalized


def psf_all_fields_one_channel(fl_zi_ims, sigproc_v2_params):
    """
    Build up a regional PSF for one channel on the RAW field-zstack images
    These images are not yet background subtracted.

    Implemented in a parallel zap over every field and then combine the
    fields into a singel z_and_region_to_psf which is

    (n_z_layers, divs, divs, peak_mea, peak_mea)

    TODO: Attach progress

    TODO: Write a test that definitely has the focus in the center of the src stack
          and make sure we get the whole zslices filled in
    """
    z_and_region_to_psf_per_field, im_focuses_per_field = zap.arrays(
        _do_psf_one_field_one_channel,
        dict(zi_ims=fl_zi_ims),
        _stack=True,
        peak_mea=sigproc_v2_params.peak_mea,
        divs=sigproc_v2_params.divs,
        n_dst_zslices=sigproc_v2_params.n_dst_zslices,
        n_src_zslices=sigproc_v2_params.n_src_zslices,
    )

    # SUM over fields
    z_and_region_to_psf = np.sum(z_and_region_to_psf_per_field, axis=0)

    z_and_region_to_psf = psf_normalize(z_and_region_to_psf)

    return z_and_region_to_psf.tolist(), im_focuses_per_field
