"""
This is the "Virtual Fluoro-sequencer". It uses Monte-Carlo simulation
to sample to distribution of dyetracks created by error modes of each
labelled peptide.

Nomenclature
    Flu
        A vector with a 0 or 1 in each position (1 means there's a dye in that channel at that position)
    n_samples
        The number of copies of the flu that are sampled
    Evolution
        The sim makes an array of n_sample copies of the flus and then modifies those along each cycle.
        evolution has shape: (n_samples, n_channels, len_of_peptide, n_cycles).
    Dye Tracks
        The dye counts are the sum along the axis=3 of evolution
    Cycles:
        There's three kinds of chemical cycles: Pres, Mocks, Edmans.
        At moment we treat pres and mocks the same in the sim
        but in reality they are different because bleaching effects are
        actually caused by both light and chemistry. We currently
        conflate these two effects into one parameters which makes
        pres and mocks essentially the same.

        The imaging happens _after_ the cycle. So, PMEEE means there are 5 images,
        ie. the first image is _after_ the first pre.
    Radiometry space:
        The brightness space in which real data from a scope lives.
        Each channel (dye) has different properties of brightess and variance.
        When the simulator runs, it produced "dyetracks" which are
        similar to radiometry except they have no noise and unit-brightness for all dyes.
    dyemat:
        A matrix form of of the dyetracks. Maybe either be 3 dimensional (n_samples, n_channels, n_cycles)
        or can be unwound into a 2-D mat like: (n_samples, n_channels * n_cycles)
    radmat:
        Similar to dyemat, but in radiometry space.
    p_*:
        The probability of an event
"""

import math
import time

import numpy as np
import pandas as pd
from plaster.run.sim_v1.sim_v1_result import (
    ArrayResult,
    DyeType,
    RadType,
    RecallType,
    SimV1Result,
)
from plaster.tools.log.log import debug
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.zap import zap
from scipy.spatial.distance import cdist


def _rand1(n_samples):
    """Mock-point"""
    return np.random.random(n_samples)


def _rand2(n_samples, flu_len):
    """Mock-point"""
    return np.random.random((n_samples, flu_len))


def _rand3(n_samples, n_channels, flu_len):
    """Mock-point"""
    return np.random.random((n_samples, n_channels, flu_len))


def _rand_lognormals(logs, sigma):
    """Mock-point"""
    return np.random.lognormal(mean=logs, sigma=sigma, size=logs.shape)


def _step_1_create_flu_and_p_bright(pep_seq_df, sim_params):
    """
    Create a flu and p_bright for the pep; both of shape (n_channels, len(peptide)).

    The flu has 1.0 wherever the peptide is labelled in that channel, zero otherwise.
    The p_bright has the correct p_bright at each position/channel.

    Note that the flu may be much longer than the sim_params.n_cycles
    but we still have to model all of the amino-acids that are past
    that point.  We call these the remainders.  They are not treated
    specially -- the evolution just adds them up like any others
    """
    n_aas = len(pep_seq_df)
    n_channels = sim_params.n_channels
    n_cycles = sim_params.n_cycles
    flu_len = max(n_cycles, n_aas)

    # EXTEND pep_seq_df out to flu_len if needed
    if n_aas < flu_len:
        n_nul = flu_len - n_aas
        nul = [pep_seq_df.pep_i.iloc[0], ".", 0]
        pep_seq_df = pep_seq_df.append(
            pd.DataFrame([nul] * n_nul, columns=pep_seq_df.columns)
        )

    # Join on the amino-acids to get labelled_pep_df.
    # Where labelled_pep_df now contains nan where the pep is unlabelled,
    # and all of the channel and label specific information in the other columns.
    # labelled_pep_df is one row per amino-acid and gets joined with various columns
    # to associate the aa-specific probabilities, etc.
    labelled_pep_df = pep_seq_df.join(
        sim_params.df.set_index("amino_acid"), on="aa", how="left"
    )

    # p_bright = is the product of (1.0 - ) all the ways the dye can fail to be visible.
    labelled_pep_df["p_bright"] = (
        (1.0 - labelled_pep_df.p_failure_to_attach_to_dye)
        * (1.0 - labelled_pep_df.p_failure_to_bind_amino_acid)
        * (1.0 - labelled_pep_df.p_non_fluorescent)
    )

    # BREAK the labelled_pep_df and labelled_p_bright_df up by channel
    # so that now each will have shape == (n_channels, len(pep_aas))
    # This is a "flu" (a Fluorosequence)
    flu = np.zeros((n_channels, flu_len))
    p_bright = np.zeros((n_channels, flu_len))
    for ch in range(n_channels):
        where_label_is_ch_mask = labelled_pep_df.ch_i.values == ch
        pos_with_this_channel = np.argwhere(where_label_is_ch_mask).flatten()

        # SET the flu[ch] == 1 where there is label in this channel
        flu[ch, pos_with_this_channel] = 1

        # SET the p_bright[ch] == p_bright where this is a label in this channel
        p_bright[ch] = np.where(
            where_label_is_ch_mask, labelled_pep_df.p_bright.values, 0
        )

    return flu, p_bright


def _step_2_initialize_samples_including_dark_sampling(flu, p_bright, n_samples):
    """
    Given flu, p_bright create the sample arrays.

    Destroys dyes according to the initial dark probability
    (that is, dyes and labels that are broken/inactive from the start)

    Returns:
        flu_samples (n_samples, n_channels, flu_len)
    """
    p_bright_samples = np.tile(p_bright, (n_samples, 1, 1))

    n_channels, flu_len = flu.shape

    # BROADCAST the flu and p_bright to samples
    flu_samples = np.tile(flu, (n_samples, 1, 1))

    # EVALUATE the darks (things that failed to bind or where the fluorophore was inactive)
    rand = _rand3(n_samples, n_channels, flu_len)

    # Note that the following compare is ">=" (which is opposite of the typical
    # destruction logic) because here we have p_bright (ie 1.0 - p_dark).
    flu_samples[rand >= p_bright_samples] = 0.0

    return flu_samples


def _step_3a_cycle_edman(curr_samples, is_mock, p_edman_failure):
    """
    Modify current_samples to simulate Edman degradation and to account for the various failure modes.

    If there was a successful Edman degradation then a np.nan is written over the
    first non-nan value in the sample.  Thus, successful Edmans make the sample
    evolve to include more np.nans in the head.

    Now, suppose this is the first Edman cycle and the current_samples looks like:
        current_samples = [
            [[0, 1, 1], [1, 0, 1]],  # Sample 0 (channel 0, 1)
            [[0, 1, 1], [1, 0, 1]],  # Sample 1 (channel 0, 1)
            ... up to n_samples
        ]

    Suppose that Edman efficiency is 0.90, so 90% of the samples will have a np.nan
    written into the first position.
    (Note: "N" in following means "np.nan")
        current_samples = [
            [[N, 1, 1], [N, 0, 1]],  # Sample 0; 1st Edman success
            [[N, 1, 1], [N, 0, 1]],  # Sample 1; 1st Edman success
            [[0, 1, 1], [1, 0, 1]],  # Sample 2; 1st Edman fail
            [[N, 1, 1], [N, 0, 1]],  # Sample 3; 1st Edman success
            ... up to n_samples
        ]

    The next cycle, the Edman's are applied to the first non-nan.
        current_samples = [
            [[N, N, 1], [N, N, 1]],  # Sample 0; 2nd Edman success
            [[N, N, 1], [N, N, 1]],  # Sample 1; 2nd Edman success
            [[N, 1, 1], [N, 0, 1]],  # Sample 2; 2nd Edman success (Note lag due to prev failure)
            [[N, 1, 1], [N, 0, 1]],  # Sample 3; 2nd Edman fail
            ... up to n_samples
        ]

    Arguments:
        curr_samples: The flu samples from the previous cycle
        is_mock: Do not apply Edman
    """
    n_samples, n_channels, flu_len = curr_samples.shape

    # BUILD a set of samples where ALL Edman succeed. (edman_success)
    # Then, using an np.where() below we will sample from the
    # two alternative the matrices (edman_success and edman_failure)
    # based on a random draw.

    edman_success = np.copy(curr_samples)  # This one we will modify with nan
    edman_failure = curr_samples

    # The first two index dimensions are simple indices that we want as is...
    sample_i, ch_i = np.mgrid[0:n_samples, 0:n_channels]

    # ...but the third index is based on the first position that isn't np.nan
    next_non_nan_cycle_i = np.argmin(np.isnan(edman_success), axis=2)

    # These three indices are used to splat np.nan onto the next non-nan cycle
    # (Note that this is NOT the same thing as edman_success[:, :, next_non_nan_cycle_i] = np.nan)
    edman_success[sample_i, ch_i, next_non_nan_cycle_i] = np.nan

    # CHOOSE selectively between those samples that succeeded and those that didn't
    # based on the p_edman_failure parameter
    if is_mock:
        # Mock and pre cycles always "fail" to edman.
        # So, set r to -1 so that the edman_success below will always fail
        r = -1 * np.ones((n_samples,))
    else:
        r = _rand1(n_samples)

    mask_edman_success = np.broadcast_to(
        r >= p_edman_failure, (flu_len, n_channels, n_samples),
    ).T

    return np.where(mask_edman_success, edman_success, edman_failure)


def _step_3b_photobleach(curr_samples, p_bleach_per_cycle_by_channel):
    """
    Photobleach the current_flus using the p_bleach_by_channel parameter
    """
    n_samples, n_channels, flu_len = curr_samples.shape
    curr_samples = np.copy(curr_samples)  # Will be modified

    for ch in range(n_channels):
        ch_samples = curr_samples[:, ch, :]

        not_nan_mask = ~np.isnan(ch_samples)

        # MASK for every position where the sample is >0 and not nan
        # i.e. locations with dyes (places allowed to photo-bleach)
        # positive_mask is an array of all False of the same shape
        positive_mask = np.full(not_nan_mask.shape, False)
        np.greater(ch_samples, 0, where=not_nan_mask, out=positive_mask)

        rand = _rand2(n_samples, flu_len)

        # MASK for all dyes that have photo-bleached in this cycle
        # by comparing the random numbers to the p_bleach_by_channel
        bleached_mask = np.full(not_nan_mask.shape, False)
        np.less(
            rand,
            p_bleach_per_cycle_by_channel[ch],
            where=positive_mask,
            out=bleached_mask,
        )

        # SPLAT a zero onto all the flu positions that bleached out
        ch_samples[bleached_mask] = 0

        # WRITE the updated channel-oriented samples onto the current_samples
        curr_samples[:, ch, :] = ch_samples

    return curr_samples


def _step_3c_detach(curr_samples, p_detach):
    """
    Detach peptides from surface by filling all channels with np.nan
    """
    n_samples, n_channels, flu_len = curr_samples.shape

    # DRAW random numbers of shape (n_samples)
    rand = _rand1(n_samples)

    detached = np.tile(
        (rand < p_detach).reshape((n_samples, 1, 1)), (1, n_channels, flu_len),
    )

    nans = np.full((n_samples, n_channels, flu_len), np.nan)
    return np.where(detached, nans, curr_samples)


def _step_3_evolve_cycles(flu_samples, sim_params):
    """
    Generate new flu samples with the various per-cycle errors such as Edman failure & bleaching for each cycle.
    This creates a new larger sample array "evolution" with an added dimension for each cycle.
    It then loops over the cycles adding progressively into the evolution array.

    Example, consider a flu with 3 cycles over 2 channels that is labelled as:
        flu = [
            [0, 1, 1],  # Channel 0: labelled at pos: [1, 2]
            [1, 0, 1],  # Channel 1: labelled at pos: [0, 2]
        ]

    Now, this flu is repeated n_samples times bringing it to 3 dimensions:
        flu_samples = [
            [[0, 1, 1], [1, 0, 1]],  # Sample 0
            [[0, 1, 1], [1, 0, 1]],  # Sample 1
            ... up to n_samples
        ]

    Arguments:
        flu_samples: ndarray of shape (n_samples, n_channels, flu_len)
        sim_params: SimParams

    """

    assert (
        sim_params.n_pres + sim_params.n_mocks >= 1
    ), "You must include at least 1 pre or mock cycle to capture the initial image"

    n_cycles = sim_params.n_cycles
    p_bleach_per_cycle_by_channel = [
        sim_params.by_channel[ch].p_bleach_per_cycle
        for ch in range(sim_params.n_channels)
    ]

    evolution = np.zeros((n_cycles, *flu_samples.shape))  # Note: now 4 dimensional

    curr_samples = flu_samples

    for cycle in range(n_cycles):
        is_pre = cycle < sim_params.n_pres
        is_mock = sim_params.n_pres <= cycle < sim_params.n_pres + sim_params.n_mocks

        if not is_pre:
            curr_samples = _step_3a_cycle_edman(
                curr_samples, is_mock, sim_params.error_model.p_edman_failure
            )
            curr_samples = _step_3c_detach(
                curr_samples, sim_params.error_model.p_detach
            )

        # SAVE into the evolution - "exposure/capture the image"
        evolution[cycle, :, :, :] = curr_samples

        # photo-bleaching is modeled as occurring *after* an exposure
        curr_samples = _step_3b_photobleach(curr_samples, p_bleach_per_cycle_by_channel)

    return evolution


def _step_4_make_dyemat(evolution):
    """
    Given the 4-D evolution array we now sum along the flu_len
    to count how many dyes there are at each (cycle, sample, channel)

    Returns:
        dyemat (n_samples, n_channels, n_cycles)
    """

    # evoluiton is: (n_cycles, n_samples, n_channels, flu_len)

    dyemat_with_nan = np.nansum(evolution, axis=3)  # Sum along the flu_len

    # dyemat_with_nan is now: (n_cycles, n_samples, n_channels)
    # But is want the n_samples first, so moveaxes
    dyemat_with_nan = np.moveaxis(dyemat_with_nan, 0, 2)

    return np.nan_to_num(dyemat_with_nan)


def _step_5_make_radmat(dyemat, sim_params):
    """
    Generate radiation for each (sample, cycle) by applying a
    log-normal radiation pattern to the dyemat.
    """

    n_samples, n_channels, n_cycles = dyemat.shape

    radiometry = np.zeros_like(dyemat)
    for ch in range(n_channels):
        log_ch_beta = math.log(sim_params.by_channel[ch].beta)
        ch_sigma = sim_params.by_channel[ch].sigma

        # dyemat can have zeros, nan these to prevent log(0)
        dm_nan = np.copy(dyemat[:, ch, :])
        dm_nan[dm_nan == 0] = np.nan

        logs = np.log(dm_nan)  # log(nan) == nan

        # Remember: log(a) + log(b) == log(a*b)
        # So we're scaling the dyes by beta and taking the log
        ch_radiometry = _rand_lognormals(log_ch_beta + logs, ch_sigma)

        radiometry[:, ch, :] = np.nan_to_num(ch_radiometry)

    return radiometry


def _step_6_compact_flu(flu, n_edmans, n_channels):
    """
    The compact_flu indicates dye locations as revealed by Edman degradation,
    and as such will be n_edmans long.  The flu length is max(n_cycles,pep_length)
    and any dyes found beyond n_edmans will contribute to the remainder_flu.
    """
    assert flu.shape[0] == n_channels and flu.shape[1] >= n_edmans
    compact_flu = np.full((n_edmans,), np.nan)
    for ch in range(n_channels):
        compact_flu[np.argwhere(flu[ch, 0:n_edmans] > 0)] = ch
    remainder_flu = np.sum(flu[:, n_edmans:], axis=1)
    return compact_flu, remainder_flu


def _do_pep_sim(
    pep_seq_df, sim_params, n_samples, output_dyemat, output_radmat, output_recall
):
    """
    Run Virtual Fluorosequencing Monte-Carlo on one peptide.

    Steps:
        1. Create the flu and p_bright matching up labels to the pep sequence
        Repeat:
            2. Create n_samples copies of the flu applying the p_bright to the initial cycle.
            3. Evolve the samples over the n_cycles cycles, creating a copy of the samples each time into a new dimension
            4. Sum up the dyes on each sample to get a dyemat
        5. Add noise, gain, variance to the dymat to make a radmat.
        6. Make a compact flu (for debugging and visualization purposes)

    Dealing with all-dark samples:
        We count but do not keep any "all-dark" samples
        because these are sampels that we can never measure
        on the actual instrument.

        In this function, we re-sample over and over until we collect
        n_samples worth of non-dark samples. But, we also keep track
        of the fraction of these all-darks for later recall
        correction.

        It could happen that a peptide is so poorly labelled
        that only a tiny fraction of the samples have signal.
        In that case we need to protect against a run-away "infinite"
        loop of re-sampling and with a limit on the number of
        allowable resamples.

    """

    """
    TFB Notes 20 April 2020

    In a next pass, I want to allow for arbitrary numbers of cycles of different kinds in
    any order, with a proposed syntax of "P20M5E10" (means 20 pre cycles, 5 mocks, 10 edmans).
    "Cycles" is a name we currently give to objects which are containers for various operations.
    In the current scheme, "pre" cycles only contain the operation "take image", which in turn
    implies modeling artifacts of that operation (photo-bleaching).

    In the current code, an order is assumed: Pres, then Mocks, then Edmans.

    In the next pass, no such order will be assumed.  If "Pre" remains the name given to
    the kind of cycle which contains only an image operation, then one could do a photobleach
    experiment with:  P100.   Or, one could take 20 images per edman with PEP20EP20EP20.
    (That's an initial Pre followed by an Edman, then 20 Pres, an Edman, etc)

    The latter gets awkward if you wanted to do 15 edmans.  So maybe we need different types
    of syntax for different use cases.  The string/count syntax is nice for many things, but
    maybe also nice is a "--n_images_per_edman" for the 15edman with 20 images each use case,
    or more generally --n_images_per_cycle.

    Or, P(EP20)15 (this is a Pre followed by 15 repeats of EP20 - each an Edman followed by 20 Pre)

    Because this is not clear to me, I want to start more simply and correct the problems
    I see with the current code.  To that end, I will ensure that

    _step_3_evolve_cycles:
        1.  We always have at least 1 pre or mock to obtain an initial image
        2.  Photo-bleaching is modeled as occurring *after* an image is taken

    _step_6_compact_flu:
        3.  The non-remainder portion of the compact flu must only be n_edmans long,
            and those 'indices' represented must be aligned with the edman cycles.

    _generate_flu_info: (part of SimV1Result)
        4. Same as (3) for the resulting fluinfo in the Result.
    """

    n_channels = sim_params.n_channels
    n_cycles = sim_params.n_cycles

    pep_i = pep_seq_df.pep_i.values[0]
    assert np.all(pep_seq_df.pep_i == pep_i)

    assert output_dyemat.shape[1] == n_samples
    assert output_radmat.shape[1] == n_samples

    # STEP 1: Get the labelled flu and p_bright
    flu, p_bright = _step_1_create_flu_and_p_bright(pep_seq_df, sim_params)

    # TODO? We once had a concept of "protecting" peptides of interest from decoys
    # when using a "shuffle" decoy_mode on peptides (MHC project).  To do this now
    # would mean examining the flus returned above to look for decoy flus that collide
    # with a "protein of interest" flu (peptides == proteins in MHC-like experiments).
    # We could reshuffle that decoy until it didn't collide, or eliminate it.  This is
    # awkward because decoy-generation is done way before label assignment so we don't
    # know about collisions until now.

    # STEPS 2-4 are run in a loop collecting non-zero rows
    n_non_dark_samples = 0
    non_dark_dyemat = np.zeros((n_samples, n_channels, n_cycles))

    recall_numer = 0
    recall_denom = 0
    n_allowable_sampling_batches = 1 if sim_params.is_survey else 10

    if sim_params.is_survey:
        assert n_samples == 1

    for iteration in range(n_allowable_sampling_batches):
        initial_flu_samples = _step_2_initialize_samples_including_dark_sampling(
            flu, p_bright, n_samples
        )

        evolution = _step_3_evolve_cycles(initial_flu_samples, sim_params)
        dyemat = _step_4_make_dyemat(evolution)

        # FIND the non-all-darks
        non_dark_rows = np.any(dyemat > 0, axis=(1, 2))
        n_non_dark_rows_in_this_batch = non_dark_rows.sum()

        recall_numer += n_non_dark_rows_in_this_batch
        recall_denom += n_samples

        # SHORTCUT out of this loop if < 1% have signal
        if n_non_dark_rows_in_this_batch < 0.01 * n_samples:
            break

        # COPY (up to n_samples) into the non_dark_samples
        n_remaining_samples_in_non_dark_samples_buffer = n_samples - n_non_dark_samples
        n_rows_to_keep = min(
            n_non_dark_rows_in_this_batch,
            n_remaining_samples_in_non_dark_samples_buffer,
        )

        non_dark_dyemat[
            n_non_dark_samples : n_non_dark_samples + n_rows_to_keep
        ] = dyemat[non_dark_rows][0:n_rows_to_keep]

        n_non_dark_samples += n_rows_to_keep

        if n_non_dark_samples >= n_samples:
            break

    if n_non_dark_samples == n_samples:
        # We got a full sampling
        output_recall[pep_i] = recall_numer / recall_denom
        output_dyemat[pep_i] = non_dark_dyemat.astype(DyeType)
        output_radmat[pep_i] = _step_5_make_radmat(non_dark_dyemat, sim_params).astype(
            RadType
        )
        # debug("non zero", pep_i, output_dyemat.shape)
    else:
        # We could not obtain a full sample after many batches,
        # declare this to be 100% dark.
        # To keep the code simpler, a zeros are returned for the dyemat, radmat
        output_recall[pep_i] = 0.0
        output_dyemat[pep_i] = np.zeros(non_dark_dyemat.shape, dtype=DyeType)
        output_radmat[pep_i] = np.zeros(non_dark_dyemat.shape, dtype=RadType)
        # debug("all zero", pep_i, output_dyemat.shape)

    compact_flu, remainder_flu = _step_6_compact_flu(
        flu, sim_params.n_edmans, sim_params.n_channels
    )

    return compact_flu, remainder_flu


def _run_sim(sim_params, pep_seqs_df, name, n_peps, n_samples, progress):
    if sim_params.get("random_seed") is not None:
        # Increment so that train and test will be different
        sim_params.random_seed += 1

    np.random.seed(sim_params.random_seed)

    dyemat = ArrayResult(
        f"{name}_dyemat",
        shape=(n_peps, n_samples, sim_params.n_channels, sim_params.n_cycles),
        dtype=DyeType,
        mode="w+",
    )
    radmat = ArrayResult(
        f"{name}_radmat",
        shape=(n_peps, n_samples, sim_params.n_channels, sim_params.n_cycles),
        dtype=RadType,
        mode="w+",
    )
    recall = ArrayResult(
        f"{name}_recall", shape=(n_peps,), dtype=RecallType, mode="w+",
    )

    flus__remainders = zap.df_groups(
        _do_pep_sim,
        pep_seqs_df.groupby("pep_i"),
        sim_params=sim_params,
        n_samples=n_samples,
        output_dyemat=dyemat,
        output_radmat=radmat,
        output_recall=recall,
        _progress=progress,
        _trap_exceptions=False,
        _process_mode=True,
    )

    flus = np.array(utils.listi(flus__remainders, 0))
    flu_remainders = np.array(utils.listi(flus__remainders, 1))

    true_pep_iz = np.repeat(np.arange(n_peps), n_samples)

    return dyemat, radmat, recall, flus, flu_remainders, true_pep_iz


def sim_v1(sim_params, prep_result, progress=None, pipeline=None):
    """
    Map the simulation over the peptides in prep_result.

    This is actually performed twice in order to get a train and (different!) test set
    The "train" set includes decoys, the test set does not; furthermore
    the the error modes and radiometry noise is different in each set.
    """

    if sim_params.random_seed is None:
        sim_params.random_seed = int(time.time())

    np.random.seed(sim_params.random_seed)

    # CREATE a *training-set* for all peptides (real and decoy)
    if pipeline:
        pipeline.set_phase(0, 2)

    # Sanity check that all the peps are accounted for
    pep_seqs_with_decoys = prep_result.pepseqs__with_decoys()
    n_peps = pep_seqs_with_decoys.pep_i.nunique()
    assert n_peps == prep_result.n_peps

    (
        train_dyemat,
        train_radmat,
        train_pep_recalls,
        train_flus,
        train_flu_remainders,
        train_true_pep_iz,
    ) = _run_sim(
        sim_params,
        pep_seqs_with_decoys,
        name="train",
        n_peps=n_peps,
        n_samples=sim_params.n_samples_train,
        progress=progress,
    )

    if sim_params.is_survey:
        test_dyemat = None
        test_radmat = None
        test_recalls = None
        test_flus = None
        test_flu_remainders = None
        test_true_pep_iz = None
    else:
        # CREATE a *test-set* for real-only peptides
        if pipeline:
            pipeline.set_phase(1, 2)

        (
            test_dyemat,
            test_radmat,
            test_recalls,
            test_flus,
            test_flu_remainders,
            test_true_pep_iz,
        ) = _run_sim(
            sim_params,
            prep_result.pepseqs__no_decoys(),
            name="test",
            n_peps=n_peps,
            n_samples=sim_params.n_samples_test,
            progress=progress,
        )

        # CHECK that the train and test are not identical in SOME non_zero_row
        # If they are, there was some sort of RNG seed errors which might happen
        # for example if sub-processes failed to re-init their RNG seeds.
        # Test this by looking at pep_i==1
        non_zero_rows = np.any(train_radmat[1] > 0, axis=(1, 2))
        non_zero_row_args = np.argwhere(non_zero_rows)[0:100]
        train_rows = train_radmat[1, non_zero_row_args].reshape(
            (
                non_zero_row_args.shape[0],
                non_zero_row_args.shape[1]
                * train_radmat.shape[2]
                * train_radmat.shape[3],
            )
        )
        test_rows = test_radmat[1, non_zero_row_args].reshape(
            (
                non_zero_row_args.shape[0],
                non_zero_row_args.shape[1]
                * test_radmat.shape[2]
                * test_radmat.shape[3],
            )
        )

        if train_rows.shape[0] > 0 and not sim_params.allow_train_test_to_be_identical:
            any_differences = np.any(np.diagonal(cdist(train_rows, test_rows)) != 0.0)
            check.affirm(any_differences, "Train and test sets are identical")

    if train_dyemat is not None:
        train_dyemat.reshape(
            (train_dyemat.shape[0] * train_dyemat.shape[1], *train_dyemat.shape[2:])
        )
    if train_radmat is not None:
        train_radmat.reshape(
            (train_radmat.shape[0] * train_radmat.shape[1], *train_radmat.shape[2:])
        )
    if test_dyemat is not None:
        test_dyemat.reshape(
            (test_dyemat.shape[0] * test_dyemat.shape[1], *test_dyemat.shape[2:])
        )
    if test_radmat is not None:
        test_radmat.reshape(
            (test_radmat.shape[0] * test_radmat.shape[1], *test_radmat.shape[2:])
        )

    # REMOVE all-zero rows (EXECPT THE FIRST which is the nul row)
    assert np.all(train_dyemat[0, :, :] == 0)
    some_non_zero_row_args = np.argwhere(
        ~np.all(train_dyemat[:, :, :] == 0, axis=(1, 2))
    ).flatten()
    some_non_zero_row_args = np.concatenate(([0], some_non_zero_row_args))

    # TASK: Plucking out the non-zero rows doesn't work well
    # with Arrtay results -- I need to rethink that.
    # For now, I'm converting this back to np.ndarray
    train_dyemat = train_dyemat[some_non_zero_row_args]
    train_radmat = train_radmat[some_non_zero_row_args]
    train_true_pep_iz = train_true_pep_iz[some_non_zero_row_args]

    if test_dyemat is not None:
        assert np.all(test_dyemat[0, :, :] == 0)
        some_non_zero_row_args = np.argwhere(
            ~np.all(test_dyemat[:, :, :] == 0, axis=(1, 2))
        ).flatten()
        # DO not add a nul row into the test data
        # some_non_zero_row_args = np.concatenate(([0], some_non_zero_row_args))
        test_dyemat = test_dyemat[some_non_zero_row_args]
        test_radmat = test_radmat[some_non_zero_row_args]
        test_true_pep_iz = test_true_pep_iz[some_non_zero_row_args]

    return SimV1Result(
        params=sim_params,
        train_dyemat=train_dyemat,
        train_radmat=train_radmat,
        train_pep_recalls=train_pep_recalls,
        train_flus=train_flus,
        train_flu_remainders=train_flu_remainders,
        train_true_pep_iz=train_true_pep_iz,
        test_dyemat=test_dyemat,
        test_radmat=test_radmat,
        test_recalls=test_recalls,
        test_flus=test_flus,
        test_true_pep_iz=test_true_pep_iz,
        test_flu_remainders=test_flu_remainders,
    )
