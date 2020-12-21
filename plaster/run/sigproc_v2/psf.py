from enum import IntEnum
import numpy as np
from plaster.run.sigproc_v2 import bg, fg
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, YX
from plaster.tools.image.imops import sub_pixel_center
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.zap import zap
from plaster.run.calib.calib import RegPSF, approximate_psf
from plaster.tools.log.log import debug


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


def psf_normalize(psf_ims):
    """
    The PSF tends to have some bias and needs to have a unit area-under-curve
    The bias is removed by fitting to a Gaussian including the offset
    and then removing the offset.
    """

    normalized_psf_ims = np.zeros_like(psf_ims)

    divs = psf_ims.shape[0]
    for y in range(divs):
        for x in range(divs):

            psf_im = psf_ims[y, x]

            if np.sum(psf_im) > 0:

                # FIT to Gaussian to get the offset
                fit_params, _ = imops.fit_gauss2(psf_im)
                bias = fit_params[6]

                psf_im = (psf_im - bias).clip(min=0)

                # NORMALIZE so that all PSF estimates have unit area-under-curve
                # The z_and_region_to_psf can have all-zero elements thus we use np_safe_divide below
                denominator = psf_im.sum()
                normalized_psf_ims[y, x] = utils.np_safe_divide(psf_im, denominator)

    return normalized_psf_ims


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
        centered_peak_im = sub_pixel_center(peak_im.astype(np.float64))

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

    if return_reasons:
        return psf, reason_masks

    return psf


def _psf_from_im(im, locs, divs=5, keep_dist=8, peak_mea=11):
    """
    Run PSF for one image.

    These are typically combined from many fields and cycles and for each channel
    to get a complete calibration.

    Arguments:
        im: One image, already background subtracted
        locs: The peak locations
        divs: Spatial divisions
        keep_dist: Pixel distance under which is considered a collision
        peak_mea: n pixel width and height to hold the peak image

    Returns:
        regional_psf
        mask_for_locs (true if a loc was accepted for use)
    """
    check.array_t(im, ndim=2)

    n_locs = locs.shape[0]
    accepted = np.zeros((n_locs,))

    # In each region gather a PSF estimate and a list of locations that were accepted.
    reg_psf_ims = np.zeros((divs, divs, peak_mea, peak_mea))
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
        reg_psf_ims[y, x] = psfs

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

        # Go backwards from local to global indexing space.
        local_accepted_iz = np.argwhere(
            reasons[:, PSFEstimateMaskFields.accepted] == 1
        ).flatten()
        local_loc_i_to_global_loc_i = np.arange(n_locs)[local_locs_mask]
        assert local_loc_i_to_global_loc_i.shape == (n_local_locs,)

        global_accepted_iz = local_loc_i_to_global_loc_i[local_accepted_iz]
        accepted[global_accepted_iz] = 1

    return reg_psf_ims, accepted > 0


def _do_psf_one_field_one_channel(cy_ims, peak_mea, divs, bandpass_kwargs):
    """
    The worker for psf_all_fields_one_channel()

    Arguments:
        cy_ims: one field, one channel. It is not yet background subtracted.
        peak_mea: size of extracted square inside of which will be the PSF sample
        divs: regional divisions
    """
    peak_dim = (peak_mea, peak_mea)

    reg_psf_ims = np.zeros((divs, divs, *peak_dim))

    approx_psf = approximate_psf()

    for cy_i, cy_im in enumerate(cy_ims):
        # Filter and find peaks on EVERY cycle because we do not know
        # which cycle a given peak turns off so we treat each cycle
        # as it own min-experiment
        filtered_im, bg_std = bg.bandpass_filter(cy_im, **bandpass_kwargs,)
        locs = fg.peak_find(filtered_im, approx_psf)

        _reg_psf_ims, _ = _psf_from_im(
            filtered_im, divs=divs, peak_mea=peak_dim[0], locs=locs
        )

        # ACCUMUATE the evidence for this cycle into the psf_ims
        reg_psf_ims = reg_psf_ims + _reg_psf_ims

    return reg_psf_ims


def psf_all_fields_one_channel(flcy_ims, sigproc_v2_params, progress=None) -> RegPSF:
    """
    Build up a regional PSF for one channel on the RAW images.

    Implemented in a parallel zap over every field and then combine the
    fields into a single RegPSF which stores: (divs, divs, peak_mea, peak_mea)

    TODO: Attach progress
    """
    check.array_t(flcy_ims, ndim=4)
    assert flcy_ims.shape[-1] == flcy_ims.shape[-2]

    region_to_psf_per_field = zap.arrays(
        _do_psf_one_field_one_channel,
        dict(cy_ims=flcy_ims),
        _stack=True,
        _progress=progress,
        peak_mea=sigproc_v2_params.peak_mea,
        divs=sigproc_v2_params.divs,
        bandpass_kwargs=dict(
            low_inflection=sigproc_v2_params.low_inflection,
            low_sharpness=sigproc_v2_params.low_sharpness,
            high_inflection=sigproc_v2_params.high_inflection,
            high_sharpness=sigproc_v2_params.high_sharpness,
        ),
    )

    # SUM over fields
    psf_ims = np.sum(region_to_psf_per_field, axis=0)
    psf_ims = psf_normalize(psf_ims)

    # At this point psf_ims is a pixel image of the PSF at each reg div.
    # ie, 4 dimensional: (divs_y, divs_x, n_pixels_y, n_pixels_w)
    # Now we convert it to Gaussian Parameters by fitting so we don't have
    # to store the pixels anymore: just the 3 critical shape parameters:
    # sigma_x, sigma_y, and rho.
    assert flcy_ims.shape[-1] == flcy_ims.shape[-2]
    reg_psf = RegPSF.from_psf_ims(flcy_ims.shape[-1], psf_ims)
    return reg_psf
