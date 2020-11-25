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
from munch import Munch
from plaster.run.sigproc_v2 import bg, fg, psf
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.run.sigproc_v2.c_sub_pixel_align.sub_pixel_align import (
    sub_pixel_align_cy_ims,
)
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.calibration.psf import Gauss2Params
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, ROI, WH, XY, YX
from plaster.tools.log.log import debug, important, prof
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plumbum import local

# Calibration
# ---------------------------------------------------------------------------------------------


def _calibrate_psf(calib, ims_import_result, sigproc_v2_params):
    """
    Arguments:
        calib:
            Where to add the calibration
        ims_import_result:
            Expects this is from a movie-based ims_import where the
            "frames" are "zstacks"
    """

    # TODO: is_movie conflates two things:
    # 1. For psf calibration, cycles don't mean chemical cycles but rather the stack of images with varying focus
    # 2. In ims_import, is_movie assumes that input files are nd2 files.
    # Once this conflation is removed, is_movie can mean just #1, and this assert can be uncommented

    # assert ims_import_result.params.is_movie is True

    focus_per_field_per_channel = []
    _, n_channels, n_zslices = ims_import_result.n_fields_channel_frames()
    for ch_i in range(0, n_channels):
        fl_zi_ims = ims_import_result.ims[:, ch_i, :]
        psf_stats_ch, focus_per_field = psf.psf_all_fields_one_channel(
            fl_zi_ims, sigproc_v2_params
        )

        prop = f"regional_psf_zstack.instrument_channel[{ch_i}]"
        calib.add({prop: psf_stats_ch})
        focus_per_field_per_channel += [focus_per_field]

    return calib, focus_per_field_per_channel


def _calibrate_illum(calib, ims_import_result, progress):
    """
    Extract a per-channel regional balance by using the foreground peaks as estimators
    """
    n_fields, n_channels, n_cycles = ims_import_result.n_fields_channel_cycles()
    fg_means = np.zeros((n_channels, ims_import_result.dim, ims_import_result.dim))
    for ch_i in range(0, n_channels):
        fl_ims = ims_import_result.ims[
            :, ch_i, 0
        ]  # Cycle 0 because it has the most peaks
        reg_bal, fg_mean = fg.fg_estimate(fl_ims, calib.psfs(ch_i), progress)
        fg_means[ch_i] = fg_mean
        assert np.all(~np.isnan(reg_bal))

        prop = f"regional_illumination_balance.instrument_channel[{ch_i}]"
        calib.add({prop: reg_bal.tolist()})

    return calib, fg_means


# Analyze Functions
# -------------------------------------------------------------------------------


def _analyze_step_1_import_balanced_images(chcy_ims, sigproc_params, calib):
    """
    Import channels and order them into the output order
    (every input channel is not necessarily used).

    Returns:
        Regionally balance and channel equalized images.
        bg_std for each channel and cycle

    Notes:
        * This is per-frame because there is a significant bleed of the foreground
          into the background. That is, more peaks in a region clearly increases the background
          and thus this needs to be done per-frame.

        * Because the background is subtracted, the returned images may contain negative values.

    TODO:
        Per-channel balance
    """
    n_channels, n_cycles = chcy_ims.shape[0:2]
    dim = chcy_ims.shape[-2:]
    dst_chcy_ims = np.zeros((n_channels, n_cycles, *dim))
    dst_chcy_ims_with_bg = np.zeros((n_channels, n_cycles, *dim))
    chcy_bg_std = np.zeros((n_channels, n_cycles))
    kernel = psf.approximate_kernel()

    # Per-frame background estimation and removal
    n_channels, n_cycles = chcy_ims.shape[0:2]
    dim = chcy_ims.shape[-2:]
    for ch_i in range(n_channels):
        reg_bal = np.array(
            calib[f"regional_illumination_balance.instrument_channel[{ch_i}]"]
        )
        assert np.all(~np.isnan(reg_bal))
        bal_im = imops.interp(reg_bal, dim)

        for cy_i in range(n_cycles):
            im = np.copy(chcy_ims[ch_i, cy_i])
            if not sigproc_params.skip_regional_balance:
                im *= bal_im

            use_low_cut_filter_method = True
            if use_low_cut_filter_method:
                # NEW low-cut technique
                # These number were hand-tuned to Abbe (512x512) and might be wrong for other
                # sizes/instruments and will need to be derivied and/or calibrated.
                falloff = 0.06
                radius = 0.080
                check.array_t(im, ndim=2, is_square=True)
                mask = 1 - imops.generate_center_weighted_tanh(im.shape[0], falloff=falloff, radius=radius)
                chcy_bg_std[ch_i, cy_i] = np.nan  # TODO
                dst_chcy_ims[ch_i, cy_i, :, :] = imops.fft_filter_with_mask(im, mask=mask)
                dst_chcy_ims_with_bg[ch_i, cy_i, :, :] = im
            else:
                # The following background_regional_estimate_im is the slowest
                # part of step_1. I made an option for skipping this
                # while developing the Gauss2 fitter where I don't need it
                if sigproc_params.skip_regional_background:
                    bg_std = np.std(im)
                    bg_median = np.median(im)
                    reg_bg = np.full((5, 5), bg_median)

                else:
                    reg_bg, reg_bg_stds = bg.background_regional_estimate_im(im, kernel)
                    bg_std = np.mean(reg_bg_stds)

                chcy_bg_std[ch_i, cy_i] = bg_std
                dst_chcy_ims[ch_i, cy_i, :, :] = bg.bg_remove(im, reg_bg)
                dst_chcy_ims_with_bg[ch_i, cy_i, :, :] = im



    return dst_chcy_ims, chcy_bg_std, dst_chcy_ims_with_bg


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

    Returns:
        aln_offsets: ndarray(n_cycles, 2); where 2 is (y, x)
    """

    kernel = psf.approximate_kernel()

    fiducial_ims = []
    for im in cy_ims:
        im = im.astype(np.float64)
        if not np.all(np.isnan(im)):
            med = float(np.nanmedian(im))
        else:
            med = 0
        im = np.nan_to_num(im, nan=med)
        fiducial_ims += [imops.convolve(im, kernel)]

    fiducial_ims = np.array(fiducial_ims) - np.median(fiducial_ims)

    noise_floor = np.percentile(fiducial_ims, 95)

    fiducial_ims = np.where(fiducial_ims < noise_floor, 0, 1).astype(np.uint8)

    # ENLARGE the points
    enlarge_radius = 3
    kernel = imops.generate_circle_mask(enlarge_radius).astype(np.uint8)
    fiducial_cy_ims = np.array(
        [cv2.dilate(im, kernel, iterations=1) for im in fiducial_ims]
    ).astype(float)

    # MASK out edge effects
    for im in fiducial_cy_ims:
        imops.edge_fill(im, 20)

    # SUB-PIXEL-ALIGN
    aln_offsets = sub_pixel_align_cy_ims(fiducial_cy_ims, slice_h=peak_mea)
    # aln_offsets = imops.align(fiducial_cy_ims)

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
    roi_dim = (roi[0].stop - roi[0].start, roi[1].stop - roi[1].start)

    aligned_chcy_ims = np.zeros((n_channels, n_cycles, *roi_dim))
    for ch_i in range(n_channels):
        for cy_i, offset in zip(range(n_cycles), aln_offsets):
            shifted_im = imops.sub_pixel_shift(chcy_ims[ch_i, cy_i], -offset)
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
    ch_mean_of_cy0_im = np.mean(chcy_ims[:, 0, :, :], axis=0)
    bg_std = np.mean(chcy_bg_stds[:, 0], axis=0)
    locs = fg.peak_find(ch_mean_of_cy0_im, kernel, bg_std)
    return locs


def _analyze_step_6_radiometry(chcy_ims, locs, calib):
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

    n_locs = len(locs)
    n_channels, n_cycles = chcy_ims.shape[0:2]

    radmat = np.full((n_locs, n_channels, n_cycles, 3), np.nan)

    for ch_i in range(n_channels):
        z_reg_psfs = calib.psfs(ch_i)
        z_reg_psfs = psf.psf_gaussianify(z_reg_psfs)

        for cy_i in range(n_cycles):
            im = chcy_ims[ch_i, cy_i]

            signal, noise, aspect_ratio = fg.radiometry_one_channel_one_cycle(
                im, z_reg_psfs, locs
            )

            radmat[:, ch_i, cy_i, 0] = signal
            radmat[:, ch_i, cy_i, 1] = noise
            radmat[:, ch_i, cy_i, 2] = aspect_ratio

    return radmat


def _analyze_step_6b_fitter(chcy_ims, locs, calib, psf_params):
    """
    Fit Gaussian.

    Arguments:
        chcy_ims: (n_channels, n_cycles, width, height)
        locs: (n_peaks, 2). The second dimension is in (y, x) order
        calib: Calibration (needed for psf)
        psf_params: The Gaussian (rho form) params for the entire PSF stack

    Returns:
        fitmat: ndarray(n_locs, n_channels, n_cycles, 3 + 8)
            Where the last dim is (sig, noi, asr) + (params of gaussian in rho form)

    """
    check.array_t(chcy_ims, ndim=4)
    check.array_t(locs, ndim=2, shape=(None, 2))

    n_locs = len(locs)
    n_channels, n_cycles = chcy_ims.shape[0:2]

    fitmat = np.full(
        (n_locs, n_channels, n_cycles, Gauss2FitParams.N_FULL_PARAMS), np.nan
    )

    for ch_i in range(n_channels):
        for cy_i in range(n_cycles):
            im = chcy_ims[ch_i, cy_i]

            params = fg.radiometry_one_channel_one_cycle_fit_method(
                im, psf_params, locs
            )

            fitmat[:, ch_i, cy_i, :] = params

    return fitmat


def _analyze_step_6c_peak_differencing(chcy_ims, locs, peak_mea):
    """
    This is an experiment based on JHD's idea of analyzing the
    distribution of differences across all pixels on a peak.
    """
    check.array_t(chcy_ims, ndim=4)
    check.array_t(locs, ndim=2, shape=(None, 2))

    n_locs = len(locs)
    n_channels, n_cycles = chcy_ims.shape[0:2]

    peak_dim = (peak_mea, peak_mea)
    peak_cy_diffs = np.full((n_locs, n_channels, n_cycles, *peak_dim), np.nan)
    peak_cys = np.full((n_locs, n_channels, n_cycles, *peak_dim), np.nan)
    peak_shifts = np.full((n_locs, n_channels, n_cycles, 2), np.nan)

    for ch_i in range(n_channels):
        for loc_i, loc in enumerate(locs):
            # Subpixel align every cycle for this peak and then difference
            cy_peak_ims = np.zeros((n_cycles, *peak_dim))
            cy_peak_shifts = np.zeros((n_cycles, 2))
            for cy_i in range(n_cycles):
                im = chcy_ims[ch_i, cy_i]

                peak_im = imops.crop(im, off=YX(loc), dim=HW(peak_dim), center=True)
                if peak_im.shape != peak_dim:
                    # Skip near edges
                    break

                if np.any(np.isnan(peak_im)):
                    # Skip nan collisions
                    break

                com_before = imops.com(peak_im ** 2)
                center_pixel = np.array(peak_im.shape) / 2
                cy_peak_ims[cy_i] = imops.sub_pixel_shift(
                    peak_im, center_pixel - com_before
                )
                cy_peak_shifts[cy_i] = center_pixel - com_before

            else:
                # DIFFERENCE only if we didn't break out (all peaks were good)
                peak_cy_diffs[loc_i, ch_i, :, :, :] = np.diff(
                    cy_peak_ims, axis=0, prepend=0
                )
                peak_cys[loc_i, ch_i, :, :, :] = cy_peak_ims
                peak_shifts[loc_i, ch_i, :, :] = cy_peak_shifts

    return peak_cy_diffs, peak_cys, peak_shifts


"""
Temporaily removed until a better metric can be established

def _analyze_step_7_filter(radmat, sigproc_v2_params, calib):
    keep_mask = np.ones((radmat.shape[0],), dtype=bool)

    n_channels, n_cycles, _ = radmat.shape
    for ch_i in range(n_channels):
        bg_std = np.min(calib[f"regional_bg_std.instrument_channel[{ch_i}]"])
        keep_mask = keep_mask | np.any(
            radmat[:, out_ch_i, :, 0] > sigproc_v2_params.sig_limit * bg_std, axis=1
        )

    if sigproc_v2_params.snr_thresh is not None:
        snr = radmat[:, :, :, 0] / radmat[:, :, :, 1]
        # Note: comparison (other than !=) in numpy of nan is always False
        keep_mask = keep_mask & np.any(snr > sigproc_v2_params.snr_thresh, axis=(1, 2))

    return keep_mask
"""


def _sigproc_analyze_field(chcy_ims, sigproc_v2_params, calib, psf_params=None):
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
        chcy_ims: from ims_import_result
        sigproc_v2_params: The SigprocParams
        calib: calibration
    """

    prof()
    debug(chcy_ims.shape)

    # Step 1: Load the images in output channel order, balance, equalize
    chcy_ims, chcy_bg_stds, chcy_ims_with_bg = _analyze_step_1_import_balanced_images(
        chcy_ims, sigproc_v2_params, calib
    )
    prof()
    # At this point, chcy_ims has its background subtracted and is
    # regionally and channel balanced. It may contain negative values.

    """
    Removed temporarily see _analyze_step_2_mask_anomalies_im for explanation
    # Step 2: Remove anomalies (at least for alignment)
    if not sigproc_v2_params.skip_anomaly_detection:
        for ch_i, cy_ims in enumerate(chcy_ims):
            chcy_ims[ch_i] = imops.stack_map(cy_ims, _analyze_step_2_mask_anomalies_im)
    """

    # Step 3: Find alignment offsets by using the mean of all channels
    # Note that this requires that the channel balancing has equalized the channel weights
    aln_offsets = _analyze_step_3_align(
        np.mean(chcy_ims, axis=0), sigproc_v2_params.peak_mea
    )
    prof()

    # Step 4: Composite with alignment
    chcy_ims = _analyze_step_4_align_stack_of_chcy_ims(chcy_ims, aln_offsets)
    prof()
    # chcy_ims is now only the shape of only intersection region so is likely
    # to be smaller than the original and not necessarily a power of 2.

    aln_offsets = np.array(aln_offsets)

    # Step 5: Peak find on combined channels
    # The goal of previous channel equalization and regional balancing is that
    # all pixels are now on an equal footing so we can now use
    # a single values for fg_thresh and bg_thresh.
    kernel = psf.approximate_kernel()
    locs = _analyze_step_5_find_peaks(chcy_ims, kernel, chcy_bg_stds)
    prof()

    # Step 6: Radiometry over each channel, cycle
    radmat = _analyze_step_6_radiometry(chcy_ims, locs, calib)
    prof()

    fitmat = None
    sftmat = None
    if sigproc_v2_params.run_analysis_gauss2_fitter:
        fitmat = _analyze_step_6b_fitter(chcy_ims, locs, calib, psf_params)
        # fitmat = _analyze_step_6b_fitter(chcy_ims_with_bg, locs, calib, psf_params)
    prof()

    difmat = None
    picmat = None
    if sigproc_v2_params.run_peak_differencing:
        difmat, picmat, sftmat = _analyze_step_6c_peak_differencing(
            chcy_ims, locs, sigproc_v2_params.peak_mea
        )
    prof()

    # Temporaily removed until a better metric can be found
    # keep_mask = _analyze_step_7_filter(radmat, sigproc_v2_params, calib)

    return (
        chcy_ims,
        locs,
        radmat,
        aln_offsets,
        fitmat,
        difmat,
        picmat,
        sftmat,
    )


def _do_sigproc_analyze_and_save_field(
    field_i, ims_import_result, sigproc_v2_params, sigproc_v2_result, calib
):
    """
    Analyze AND SAVE one field by calling the sigproc_v2_result.save_field()
    """

    chcy_ims = ims_import_result.ims[field_i]
    n_channels, n_cycles, roi_h, roi_w = chcy_ims.shape

    psf_params = None
    if sigproc_v2_params.run_psf_gauss2_fitter:
        psf_params = psf.psf_fit_gaussian(calib.psfs(0))

    (
        chcy_ims,
        locs,
        radmat,
        aln_offsets,
        fitmat,
        difmat,
        picmat,
        sftmat,
    ) = _sigproc_analyze_field(chcy_ims, sigproc_v2_params, calib, psf_params)

    mea = np.array([chcy_ims.shape[-1:]])
    if np.any(aln_offsets ** 2 > (mea * 0.1) ** 2):
        important(f"field {field_i} has bad alignment {aln_offsets}")

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
            )
            for channel_i in range(n_channels)
            for cycle_i in range(n_cycles)
        ],
        columns=list(SigprocV2Result.field_df_schema.keys()),
    )

    assert len(radmat) == len(peak_df)

    debug(field_df)

    sigproc_v2_result.save_field(
        field_i,
        peak_df=peak_df,
        field_df=field_df,
        radmat=radmat,
        fitmat=fitmat,
        difmat=difmat,
        picmat=picmat,
        sftmat=sftmat,
        _aln_chcy_ims=chcy_ims,
    )

    debug()


# Entrypoints
# -------------------------------------------------------------------------------


def sigproc_instrument_calib(sigproc_v2_params, ims_import_result, progress=None):
    """
    Entrypoint for PSF calibration.
    Requires a z-stack movie

    TODO:
        Progress
    """

    focus_per_field_per_channel = None
    calib = None
    fg_means = None

    if sigproc_v2_params.mode == common.SIGPROC_V2_PSF_CALIB:
        calib = Calibration()
        calib, focus_per_field_per_channel = _calibrate_psf(
            calib, ims_import_result, sigproc_v2_params
        )

    elif sigproc_v2_params.mode == common.SIGPROC_V2_ILLUM_CALIB:
        if not local.path(sigproc_v2_params.calibration_file).exists():
            raise FileNotFoundError(
                "SIGPROC_V2_ILLUM_CALIB requires an existing calibration file"
            )

        calib = Calibration.load(sigproc_v2_params.calibration_file)
        calib, fg_means = _calibrate_illum(calib, ims_import_result, progress,)

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

    if calib is None:
        calib = Calibration.load(sigproc_v2_params.calibration_file)

    assert not calib.is_empty()

    sigproc_v2_result = SigprocV2Result(
        params=sigproc_v2_params,
        n_channels=ims_import_result.n_channels,
        n_cycles=ims_import_result.n_cycles,
        calib=calib,
        focus_per_field_per_channel=None,
    )

    n_fields = ims_import_result.n_fields
    n_fields_limit = sigproc_v2_params.n_fields_limit
    if n_fields_limit is not None and n_fields_limit < n_fields:
        n_fields = n_fields_limit

    debug(n_fields)

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
        ],
        _trap_exceptions=False,
        _progress=progress,
    )

    return sigproc_v2_result
