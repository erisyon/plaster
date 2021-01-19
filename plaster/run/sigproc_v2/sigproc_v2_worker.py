"""
This is the Signal Processor that extracts data from images from the fluoro-sequencer microscope.

Nomenclature
    Field
        One position of the X/Y stage
    Channel
        One wavelength of measured light
    Cycle
        One chemical cycle (Pre, Mock or Edman)
    Anomaly
        An area of an image that has a problem (dust, etc)
    Raw image
        Unmodified images from the scope
    Regional
        When a parameter is varies spatially
    Balance image
        A raw image scaled to compensate for regional uneven illumination
        and for differences between channels.
    Aligned field stack
        The scope stage is not perfect and does not return to exactly the same position
        each cycle, a computational alignment correction finds the optimal X/Y translation.
    ROI
        A Region Of Interest
    Intersection ROI
        In an aligned field stack, the Intersection ROI is the set
        of pixels that are in every cycle.  Ie, typically smaller than the
        dimensions of the raw images.
    Composite image
        When one or more of the channels/cycles for a field are stacked
    Fiducial images
        An image that is intended only to enhance the alignment or
        peak finding algorithm. These images are temporary and discarded after use.
    Peak/Loc/Spot
        A Peak, LOC-action, or Spot found in the image that presumably
        is generated by a single molecule.
    Radmat (aka "Radiometry Matrix")
        A matrix such that each row is a peak and each column is a measurement of brightness
        for each channel/cycle.
        Sometimes stored in (n_peaks, n_channels, n_cycles)
        Sometimes stored flatten as (n_peaks, n_channels * n_cycles)
    Radrow
        A single row (cooresponding to a single peak) of a radmat.
    cy_ims
        A set of images through all cycles for one field/channel.
    chcy_ims
        A set of images for all channel/cycles for one field.
    flchcy_ims
        A set of images for all field/channel/cycles.


Calibration-Related Components
    Sigproc Calibration is a notebook activity until we can automated it well.
    It records a Calibration object that contains:
        * regional_illumination_balance
        * regional_bg_mean
        * regional_bg_std
        * regional_psf_zstack
        * zstack_depths

V2 flow:
    0. Load calibration
        Compare the subject-id, brightness settings in the tsv files
        with what was in the calibration.
    1. Import balanced images
        Re-orders images in to output channel order
        Regionally balances images given calib
            (subtract background and scale regionally by the balance map)
        Channel equalize
            (Scale so all channels are the same strength)
    2. Mask anomalies
        Write nan into anomalies
    3. Align cycles
        Finds translations per cycles to align with cycle 0
    4. Composite with alignment offsets
        Discards pixels that are not in every cycle.
    5. Find peaks
    6. Radiometry
    7. Remove empties


TASKS:
    * Tune the size of the kernels used (1.5)
        Also, the peak_find has a similar hard-coded value

    * A general "anomaly" report section would be useful
        Frames that were very bad
        Frames where the SNR is really bad
        General histograms of quality, anomalies, and SNR

    * Examine each cycle and fit best z-depth of the PSFs
      and use that for the radiometry of that cycle

    * _compute_channel_weights needs to change over to
      calibration-time computation.

"""

import cv2
import numpy as np
import pandas as pd
import plaster.run.sigproc_v2.psf
import plaster.run.sigproc_v2.reg_psf
from munch import Munch
from plaster.run.calib.calib import Calib
from plaster.run.sigproc_v2 import bg, fg, psf
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.run.sigproc_v2.c_sub_pixel_align.sub_pixel_align import (
    sub_pixel_align_cy_ims,
)
from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import Gauss2FitParams
from plaster.run.calib.calib import Calib, RegPSF, RegIllum, CalibIdentity
from plaster.tools.image import imops
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plaster.tools.utils import utils
from plaster.run.sigproc_v2.c_radiometry.radiometry import radiometry_field_stack
from plaster.run.sigproc_v2.c_radiometry import radiometry
from plaster.run.sigproc_v2.c_gauss2_fitter import gauss2_fitter
from plaster.run.sigproc_v2.c_sub_pixel_align import sub_pixel_align
from plaster.tools.log.log import debug, important, prof, exception

# Calibration
# ---------------------------------------------------------------------------------------------


def _calibrate(ims_import_result, sigproc_v2_params, progress):
    """
    Extract a PSF and extract illumination balance from (assumed) 1-count data.

    Arguments:
        TODO

    Returns:
        calib, with new records added
        fg_means:
    """

    calib = Calib()

    if sigproc_v2_params.n_fields_limit is None:
        # Use quality metrics
        q = ims_import_result.qualities()
        q_by_field = q.groupby("field_i").quality.mean()
        med_field_q = np.median(q_by_field.values)
        good_field_iz = q_by_field[q_by_field > med_field_q].index.values
        flchcy_ims = ims_import_result.ims[:, :, :].astype(np.float64)
        flchcy_ims = flchcy_ims[good_field_iz]
    else:
        field_slice = slice(0, sigproc_v2_params.n_fields_limit, 1)
        flchcy_ims = ims_import_result.ims[field_slice, :, :].astype(np.float64)

    n_channels = ims_import_result.n_channels
    ch_reg_psfs = []
    for ch_i in range(0, n_channels):
        flcy_ims = flchcy_ims[:, ch_i, :]
        ch_reg_psfs += [
            psf.psf_all_fields_one_channel(flcy_ims, sigproc_v2_params, progress)
        ]

    reg_psf = RegPSF.from_channel_reg_psfs(ch_reg_psfs)
    calib.add_reg_psf(reg_psf)

    bandpass_kwargs = dict(
        low_inflection=sigproc_v2_params.low_inflection,
        low_sharpness=sigproc_v2_params.low_sharpness,
        high_inflection=sigproc_v2_params.high_inflection,
        high_sharpness=sigproc_v2_params.high_sharpness,
    )

    fg_means = np.zeros((n_channels, ims_import_result.dim, ims_import_result.dim))
    for ch_i in range(0, n_channels):
        # TODO: Eventually I'd like to change this so that it uses all cycles
        # But that means some refactoring on fg_estimate(); see the notes about
        # the zap and accumulation buffers. Until then this runs only on cycle 0
        cy_i = 0
        fl_ims = flchcy_ims[:, ch_i, cy_i]
        reg_bal, fg_mean = fg.fg_estimate(fl_ims, calib.reg_psf(ch_i), bandpass_kwargs)
        fg_means[ch_i] = fg_mean
        check.array_t(reg_bal, ndim=2)
        assert np.all(~np.isnan(reg_bal))
        assert reg_bal.shape[-1] == reg_bal.shape[-2]
        assert reg_bal.shape[0] == reg_bal.shape[1]
        assert fl_ims.shape[-1] == fl_ims.shape[-2]
        reg_illum = RegIllum(
            n_channels=n_channels, im_mea=fl_ims.shape[-1], n_divs=reg_bal.shape[0]
        )
        reg_illum.set(ch_i, reg_bal)
        calib.add_reg_illum(reg_illum)

    return calib, fg_means


# Analyze Functions
# -------------------------------------------------------------------------------


def _analyze_step_1_import_balanced_images(chcy_ims, sigproc_params, calib):
    """
    Import channels and order them into the output order
    (every input channel is not necessarily used).

    Returns:
        dst_filtered_chcy_ims: Balanced and band-pass filtered
        dst_chcy_bg_std: Std on the
        dst_unfiltered_chcy_ims: Balanced but not band-pass filtered

    Notes:
        * Because the background is subtracted, the returned images may contain negative values.

    TODO:
        Per-channel balance
    """
    n_channels, n_cycles = chcy_ims.shape[0:2]
    dim = chcy_ims.shape[-2:]
    dst_filt_chcy_ims = np.zeros((n_channels, n_cycles, *dim))
    dst_unfilt_chcy_ims = np.zeros((n_channels, n_cycles, *dim))
    dst_chcy_bg_std = np.zeros((n_channels, n_cycles))
    approx_psf = plaster.run.calib.calib.approximate_psf()

    # Per-frame background estimation and removal
    n_channels, n_cycles = chcy_ims.shape[0:2]
    dim = chcy_ims.shape[-2:]
    for ch_i in range(n_channels):
        bal_im = calib.reg_illum().interp(ch_i)

        assert bal_im.sum() > 0.0, "Sanity check"

        for cy_i in range(n_cycles):
            im = np.copy(chcy_ims[ch_i, cy_i])
            if sigproc_params.run_regional_balance:
                im *= bal_im

            if sigproc_params.run_bandpass_filter:
                filtered_im, bg_std = bg.bandpass_filter(
                    im,
                    low_inflection=sigproc_params.low_inflection,
                    low_sharpness=sigproc_params.low_sharpness,
                    high_inflection=sigproc_params.high_inflection,
                    high_sharpness=sigproc_params.high_sharpness,
                )

            else:
                filtered_im, bg_std = bg.bg_estimate_and_remove(im, approx_psf,)

            dst_unfilt_chcy_ims[ch_i, cy_i, :, :] = im
            dst_filt_chcy_ims[ch_i, cy_i, :, :] = filtered_im
            dst_chcy_bg_std[ch_i, cy_i] = bg_std

    return dst_filt_chcy_ims, dst_unfilt_chcy_ims, dst_chcy_bg_std


'''
Removed temporarily because this function needs signficant tuning
and for now it is easier to remove whole fields instead of partial fields
Also, this should be moved to FG or BG as it is really a helper

def _analyze_step_2_mask_anomalies_im(im, den_threshold=300):
    """
    Operates on pre-balanced images.
    The den_threshold of 300 was found empirically on Val data

    Sets anomalies to nan
    """
    import skimage.transform  # Defer slow imports
    import cv2

    check.array_t(im, is_square=True)

    # SLICE into square using numpy-foo by reshaping the image
    # into a four-dimensional array can then by np.mean on the inner dimensions.
    sub_mea = 4  # Size of the sub-sample region
    im_mea, _ = im.shape

    squares = im.reshape(im_mea // sub_mea, sub_mea, im_mea // sub_mea, sub_mea)
    # At this point, im is now 4-dimensional like: (256, 2, 256, 2)
    # But we want the small_dims next to each other for simplicity so swap the inner axes
    squares = squares.swapaxes(1, 2)
    # Now squares is (256, 256, 2, 2.)

    # squares is like: 256, 256, 2, 2. So we need the mean of the last two axes
    squares = np.mean(squares, axis=(2, 3))

    bad_mask = (squares > den_threshold).astype(float)

    # EXPAND the bad areas by erosion and dilate.
    # Erosion gets rid of the single-pixel hits and dilation expands the bad areas
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(bad_mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)

    scale = im.shape[0] // mask.shape[0]

    full_size_mask = skimage.transform.rescale(
        mask, scale=scale, multichannel=False, mode="constant", anti_aliasing=False
    ).astype(bool)

    # FIND rect contours of bad areas
    contours, hierarchy = cv2.findContours(
        full_size_mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    bad_rects = [cv2.boundingRect(cnt) for cnt in contours]
    im = im.copy()
    for rect in bad_rects:
        imops.fill(im, loc=XY(rect[0], rect[1]), dim=WH(rect[2], rect[3]), val=np.nan)

    return im
'''


def _analyze_step_3_align(cy_ims, peak_mea):
    """
    Align a stack of cy_ims by generating simplified fiducials for each cycle
    (assumes camera does not move between channels)

    TODO: Convert this to a gradient descent using fft_shifting which
          will allow deprecation of the c_sub_pixel_align module.

    Returns:
        aln_offsets: ndarray(n_cycles, 2); where 2 is (y, x)
    """

    approx_psf = plaster.run.calib.calib.approximate_psf()

    fiducial_ims = []
    for im in cy_ims:
        im = im.astype(np.float64)
        if not np.all(np.isnan(im)):
            med = float(np.nanmedian(im))
        else:
            med = 0
        im = np.nan_to_num(im, nan=med)
        fiducial_ims += [imops.convolve(im, approx_psf)]

    fiducial_ims = np.array(fiducial_ims) - np.median(fiducial_ims)

    noise_floor = np.percentile(fiducial_ims, 95)

    fiducial_ims = np.where(fiducial_ims < noise_floor, 0, 1).astype(np.uint8)

    # ENLARGE the points
    enlarge_radius = 3
    approx_psf = imops.generate_circle_mask(enlarge_radius).astype(np.uint8)
    fiducial_cy_ims = np.array(
        [cv2.dilate(im, approx_psf, iterations=2) for im in fiducial_ims]
    ).astype(float)

    # MASK out edge effects
    for im in fiducial_cy_ims:
        imops.edge_fill(im, 20)

    # APPLY fiducial_cy_ims as a mask over real data
    cy_ims = np.where(fiducial_ims > 0, cy_ims, 0)

    # SUB-PIXEL-ALIGN
    aln_offsets = sub_pixel_align_cy_ims(cy_ims, slice_h=peak_mea)

    return aln_offsets


def _analyze_step_4_align_stack_of_chcy_ims(chcy_ims, aln_offsets):
    """
    Given the alignment_offsets, create a new image stack that
    has the dimensions of the intersection ROI (ie the overlapping
    region that contains pixels from all cycles)

    Returns:
        A newly allocated ndarray(n_channels, n_cycles, ROI)
        where ROI is the region of interest determined by the
        pixels that all cycles ahve in common

    Notes:
        The returned image is likely smaller than the chcy_ims shape.
    """
    check.array_t(chcy_ims, ndim=4)
    n_channels, n_cycles = chcy_ims.shape[0:2]
    check.array_t(aln_offsets, shape=(n_cycles, 2))
    assert n_cycles == aln_offsets.shape[0]
    chcy_ims = chcy_ims.astype(np.float64)

    raw_dim = chcy_ims.shape[-2:]
    roi = imops.intersection_roi_from_aln_offsets(aln_offsets, raw_dim)
    roi_dim = (
        roi[0].stop - roi[0].start,
        roi[1].stop - roi[1].start,
    )

    aligned_chcy_ims = np.zeros((n_channels, n_cycles, *roi_dim))

    for ch_i in range(n_channels):
        for cy_i, offset in zip(range(n_cycles), aln_offsets):

            # Sub-pixel shift the square raw images using phase shifting
            # (This must be done with square images)
            im = chcy_ims[ch_i, cy_i]
            shifted_im = imops.fft_sub_pixel_shift(im, -offset)

            # Now that it is shifted we pluck out the ROI into the destination
            aligned_chcy_ims[ch_i, cy_i, 0 : roi_dim[0], 0 : roi_dim[1]] = shifted_im[
                roi[0], roi[1]
            ]

    return aligned_chcy_ims


def _analyze_step_5_find_peaks(chcy_ims, kernel, chcy_bg_stds):
    """
    Step 5: Peak find on combined channels

    The goal of previous channel equalization and regional balancing is that
    all pixels are now on an equal footing so we can now use
    a single values for fg_thresh and bg_thresh.

    Returns:
        locs: ndarray (n_peaks, 2)  where the second dimaension is in y, x order
    """

    # Use more than one cycle to improve the quality of the sub-pixel estimate
    # But then discard peaks that are off after cycle 1??

    # TODO: Remove bg_std and derive it now that I have cleaner bg subtraction with band pass

    ch_mean_of_cy0_im = np.mean(chcy_ims[:, 0, :, :], axis=0)
    # bg_std = np.mean(chcy_bg_stds[:, 0], axis=0)
    try:
        locs = fg.sub_pixel_peak_find(ch_mean_of_cy0_im, kernel)
    except Exception as e:
        exception(e, "Failure during peak find, no peaks recorded for this frame.")
        locs = np.zeros((0, 2))
    return locs


def _analyze_step_6a_fitter(
    chcy_ims, locs, reg_psf: plaster.run.calib.calib.RegPSF, mask
):
    """
    Fit Gaussian.

    Arguments:
        chcy_ims: (n_channels, n_cycles, width, height)
        locs: (n_peaks, 2). The second dimension is in (y, x) order
        calib: Calibration (needed for psf)
        psf_params: The Gaussian (rho form) params for the entire PSF stack
        mask: Used to subsample the locs (where true)

    Returns:
        fitmat: ndarray(n_locs, n_channels, n_cycles, 3 + 8)
            Where the last dim is (sig, noi, asr) + (params of gaussian in rho form)

    """
    check.array_t(chcy_ims, ndim=4)
    check.array_t(locs, ndim=2, shape=(None, 2))
    check.array_t(mask, shape=(locs.shape[0],), dtype=bool)

    n_locs = len(locs)
    n_channels, n_cycles = chcy_ims.shape[0:2]

    fitmat = np.full(
        (n_locs, n_channels, n_cycles, Gauss2FitParams.N_FULL_PARAMS), np.nan
    )

    # The radiometry_one_channel_one_cycle_fit_method is build to skip any loc that
    # is NaN so that is how we limit with the mask.
    locs = locs.copy()
    locs[~mask, :] = np.nan

    for ch_i in range(n_channels):
        for cy_i in range(n_cycles):
            im = chcy_ims[ch_i, cy_i]
            params = fg.radiometry_one_channel_one_cycle_fit_method(im, reg_psf, locs)
            fitmat[:, ch_i, cy_i, :] = params

    return fitmat


def _analyze_step_6b_radiometry(chcy_ims, locs, calib, focus_adjustment):
    """
    Extract radiometry (signal and noise) from the field chcy stack.

    Arguments:
        chcy_ims: (n_channels, n_cycles, width, height)
        locs: (n_peaks, 2). The second dimension is in (y, x) order
        calib: Calibration (needed for psf)

    Returns:
        radmat: ndarray(n_locs, n_channels, n_cycles, 3)
            Where the last dim is (signal, noise, aspect_ratio)
    """
    check.array_t(chcy_ims, ndim=4)
    check.array_t(locs, ndim=2, shape=(None, 2))

    n_channels, n_cycles = chcy_ims.shape[0:2]
    assert (
        n_channels == 1
    ), "Until further notice, only passing in one reg_psf for one channel"
    reg_psf = calib.reg_psf(0)

    radmat = radiometry_field_stack(
        chcy_ims, locs=locs, reg_psf=reg_psf, focus_adjustment=focus_adjustment
    )

    return radmat


def _sigproc_analyze_field(
    filt_chcy_ims,
    sigproc_v2_params,
    calib,
    reg_psf: plaster.run.calib.calib.RegPSF = None,
):
    """
    Analyze one field --
        * Regional and channel balance
        * remove anomalies (temporarily removed)
        * Align cycles
        * Composite aligned
        * Peak find
        * Radiometry
        * Filtering (temporarily removed)

    Arguments:
        filt_chcy_ims: from ims_import_result
        sigproc_v2_params: The SigprocParams
        calib: calibration
    """

    # Step 1: Load the images in output channel order, balance, equalize
    # Timings:
    #   Val8_2t: 20 seconds per field, using a single core, probably IO bound
    (
        filt_chcy_ims,
        unfilt_chcy_ims,
        chcy_bg_stds,
    ) = _analyze_step_1_import_balanced_images(
        filt_chcy_ims.astype(np.float64), sigproc_v2_params, calib
    )

    n_cycles = filt_chcy_ims.shape[1]

    """
    Removed temporarily see _analyze_step_2_mask_anomalies_im for explanation
    # Step 2: Remove anomalies (at least for alignment)
    if sigproc_v2_params.run_anomaly_detection:
        for ch_i, cy_ims in enumerate(chcy_ims):
            chcy_ims[ch_i] = imops.stack_map(cy_ims, _analyze_step_2_mask_anomalies_im)
    """

    # Step 3: Find alignment offsets by using the mean of all channels
    # Note that this requires that the channel balancing has equalized the channel weights
    # Timings:
    #   Val8_2t: 53 seconds per field, single core for about 20 seconds and then
    #            several bursts of all cores. Presumably that early delay is load time
    if sigproc_v2_params.run_aligner:
        aln_offsets = _analyze_step_3_align(
            np.mean(filt_chcy_ims, axis=0), sigproc_v2_params.peak_mea
        )
    else:
        aln_offsets = np.zeros((n_cycles, 2))

    # Step 4: Composite with alignment
    # Timings:
    #   Val8_2t: Each of the following two taking 14 sec (28 sec combined)
    #            Completely single core. This could probably be parallelized somehow
    aln_filt_chcy_ims = _analyze_step_4_align_stack_of_chcy_ims(
        filt_chcy_ims, aln_offsets
    )
    aln_unfilt_chcy_ims = _analyze_step_4_align_stack_of_chcy_ims(
        unfilt_chcy_ims, aln_offsets
    )

    # aln_*filt_chcy_ims is now only the shape of only intersection region so is likely
    # to be smaller than the original and not necessarily a power of 2.

    aln_offsets = np.array(aln_offsets)

    # Step 5: Peak find on combined channels
    # The goal of previous channel equalization and regional balancing is that
    # all pixels are now on an equal footing so we can now use
    # a single values for fg_thresh and bg_thresh.

    if sigproc_v2_params.locs is not None:
        locs = np.array(sigproc_v2_params.locs)
        assert locs.ndim == 1
        locs = np.reshape(locs, (locs.shape[0] // 2, 2))
    else:
        approx_psf = plaster.run.calib.calib.approximate_psf()
        locs = _analyze_step_5_find_peaks(aln_filt_chcy_ims, approx_psf, chcy_bg_stds)

    n_locs = len(locs)

    # Step 6: Radiometry over each channel, cycle

    # Sample all or a sub-set of peaks for focus purposes (and also for debugging)
    if sigproc_v2_params.run_analysis_gauss2_fitter:
        mask = np.ones((n_locs,), dtype=bool)

    else:
        # Subsample peaks for fitting
        mask = np.zeros((n_locs,), dtype=bool)
        count = 100  # Don't know if this is enough
        if n_locs > 0:
            try:
                iz = np.random.choice(
                    n_locs, count
                )  # Allow replace in case count > n_locs
                mask[iz] = 1
            except ValueError:
                pass

    fitmat = _analyze_step_6a_fitter(aln_unfilt_chcy_ims, locs, reg_psf, mask)

    # At moment it appears that focus adjustment does nothing under the
    # the filters and alignment -- because there is no correlation anymore
    # between peak width and brightness, so for now I'm turning off the
    # focus adjustment but leacing in the fitmat sampling for reporting purposes

    # if sigproc_v2_params.run_focal_adjustments:
    #     focus_adjustments = fg.focus_from_fitmat(fitmat, reg_psf)
    # else:
    #     focus_adjustments = np.ones((n_cycles,))

    focus_adjustments = np.ones((n_cycles,))

    # This is taking about 2.5 seconds (0.5 of which is the interpolation of the psf)
    # Seems surprisingly slow. Something is up.
    radmat = _analyze_step_6b_radiometry(
        aln_filt_chcy_ims, locs, calib, focus_adjustments
    )

    neighborhood_stats = None
    if sigproc_v2_params.run_neighbor_stats:
        from plaster.tools.image.coord import WH, XY

        sub_mea = 31
        peak_mea = 11
        bot = sub_mea // 2 - peak_mea // 2
        top = bot + peak_mea
        lft = sub_mea // 2 - peak_mea // 2
        rgt = bot + peak_mea
        ch_i = 0
        neighborhood_stats = np.zeros((n_locs, n_cycles, 4))
        with utils.np_no_warn():
            for loc_i, (y, x) in enumerate(locs):
                x = int(x + 0.5)
                y = int(y + 0.5)
                for cy_i in range(n_cycles):
                    sub_im = imops.crop(
                        aln_filt_chcy_ims[ch_i, cy_i],
                        off=XY(x, y),
                        dim=WH(32, 32),
                        center=True,
                    )
                    sub_im[bot:top, lft:rgt] = np.nan
                    neighborhood_stats[loc_i, cy_i, 0] = np.nanmean(sub_im)
                    neighborhood_stats[loc_i, cy_i, 1] = np.nanstd(sub_im)
                    neighborhood_stats[loc_i, cy_i, 2] = np.nanmedian(sub_im)
                    a = np.nanpercentile(sub_im, [75, 25])
                    try:
                        neighborhood_stats[loc_i, cy_i, 3] = a[0] - a[1]
                    except IndexError as e:
                        neighborhood_stats[loc_i, cy_i, 3] = np.nan

    return (
        aln_filt_chcy_ims,
        aln_unfilt_chcy_ims,
        locs,
        radmat,
        aln_offsets,
        fitmat,
        focus_adjustments,
        neighborhood_stats,
    )


def _do_sigproc_analyze_and_save_field(
    field_i, ims_import_result, sigproc_v2_params, sigproc_v2_result, calib
):
    """
    Analyze AND SAVE one field by calling the sigproc_v2_result.save_field()
    """

    chcy_ims = ims_import_result.ims[field_i]
    n_channels, n_cycles, roi_h, roi_w = chcy_ims.shape

    assert n_channels == 1

    reg_psf = calib.reg_psf()
    reg_psf.select_ch(0)  # TODO: Multichannel

    (
        aln_filt_chcy_ims,
        aln_unfilt_chcy_ims,
        locs,
        radmat,
        aln_offsets,
        fitmat,
        focus_adjustments,
        neighborhood_stats,
    ) = _sigproc_analyze_field(chcy_ims, sigproc_v2_params, calib, reg_psf)

    mea = np.array([chcy_ims.shape[-1:]])
    bad_align_mask = aln_offsets ** 2 > (mea * 0.2) ** 2
    if np.any(bad_align_mask):
        important(
            f"field {field_i} has bad alignment @ {np.argwhere(bad_align_mask)} {aln_offsets[bad_align_mask]}"
        )

    # Assign 0 to "peak_i" in the following DF because that is the GLOBAL peak_i
    # which is not computable until all fields are processed. It will be fixed up later
    # by the SigprocV2Result helper methods
    peak_df = pd.DataFrame(
        [(0, field_i, peak_i, loc[0], loc[1]) for peak_i, loc in enumerate(locs)],
        columns=list(SigprocV2Result.peak_df_schema.keys()),
    )

    field_df = pd.DataFrame(
        [
            (
                field_i,
                channel_i,
                cycle_i,
                aln_offsets[cycle_i, 0],
                aln_offsets[cycle_i, 1],
                focus_adjustments[cycle_i],
            )
            for channel_i in range(n_channels)
            for cycle_i in range(n_cycles)
        ],
        columns=list(SigprocV2Result.field_df_schema.keys()),
    )

    assert len(radmat) == len(peak_df)

    sigproc_v2_result.save_field(
        field_i,
        peak_df=peak_df,
        field_df=field_df,
        radmat=radmat,
        fitmat=fitmat,
        neighborhood_stats=neighborhood_stats,
        _aln_filt_chcy_ims=aln_filt_chcy_ims,
        _aln_unfilt_chcy_ims=aln_unfilt_chcy_ims,
    )


# Entrypoints
# -------------------------------------------------------------------------------


def sigproc_instrument_calib(sigproc_v2_params, ims_import_result, progress=None):
    """
    Entrypoint for Illumination and PSF calibration.
    """

    radiometry.init()
    gauss2_fitter.init()
    sub_pixel_align.init()

    focus_per_field_per_channel = None
    calib = None
    fg_means = None

    if sigproc_v2_params.mode == common.SIGPROC_V2_ILLUM_CALIB:
        calib, fg_means = _calibrate(ims_import_result, sigproc_v2_params, progress)

    return SigprocV2Result(
        params=sigproc_v2_params,
        n_channels=None,
        n_cycles=None,
        channel_weights=None,
        calib=calib,
        focus_per_field_per_channel=focus_per_field_per_channel,
        _fg_means=fg_means,
    )


def sigproc_analyze(sigproc_v2_params, ims_import_result, progress, calib=None):
    """
    Entrypoint for analysis of (ie generate radiometry).
    Requires a calibration_file previously generated by sigproc_instrument_calib()
    that is refered to in sigproc_v2_params.calibration_file

    If calib is not None it over-rides the loading of a calibration_file
    (used for testing)
    """

    radiometry.init()
    gauss2_fitter.init()
    sub_pixel_align.init()

    if sigproc_v2_params.no_calib:
        assert sigproc_v2_params.instrument_identity is None
        assert (
            sigproc_v2_params.no_calib_psf_sigma is not None
        ), "In no_calib mode you must specify an estimated no_calib_psf_sigma"
        calib_identity = CalibIdentity("_identity")

        calib = Calib()
        arr = np.zeros((ims_import_result.n_channels, 5, 5, RegPSF.N_PARAMS))
        arr[:, :, :, RegPSF.SIGMA_X] = sigproc_v2_params.no_calib_psf_sigma
        arr[:, :, :, RegPSF.SIGMA_Y] = sigproc_v2_params.no_calib_psf_sigma
        arr[:, :, :, RegPSF.RHO] = 0.0
        reg_psf = RegPSF.from_array(im_mea=ims_import_result.dim, peak_mea=11, arr=arr)
        calib.add_reg_psf(reg_psf, calib_identity)

        reg_illum = RegIllum.identity(
            ims_import_result.n_channels, ims_import_result.dim
        )
        calib.add_reg_illum(reg_illum, calib_identity)

    elif calib is None:
        assert sigproc_v2_params.instrument_identity is not None
        calib = Calib.load_file(
            sigproc_v2_params.calibration_file, sigproc_v2_params.instrument_identity
        )
        assert calib.has_records()

    n_fields = ims_import_result.n_fields
    n_fields_limit = sigproc_v2_params.n_fields_limit
    if n_fields_limit is not None and n_fields_limit < n_fields:
        n_fields = n_fields_limit

    sigproc_v2_result = SigprocV2Result(
        params=sigproc_v2_params,
        n_fields=n_fields,
        n_channels=ims_import_result.n_channels,
        n_cycles=ims_import_result.n_cycles,
        calib=calib,
        focus_per_field_per_channel=None,
    )

    with zap.Context(trap_exceptions=False, progress=progress):
        zap.work_orders(
            [
                Munch(
                    fn=_do_sigproc_analyze_and_save_field,
                    field_i=field_i,
                    ims_import_result=ims_import_result,
                    sigproc_v2_params=sigproc_v2_params,
                    sigproc_v2_result=sigproc_v2_result,
                    calib=calib,
                )
                for field_i in range(n_fields)
            ]
        )

    sigproc_v2_result.save()

    return sigproc_v2_result
