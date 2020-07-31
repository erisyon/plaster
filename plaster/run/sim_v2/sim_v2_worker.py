"""
This is the "Virtual Fluoro-Sequencer". It uses Monte-Carlo simulation
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
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from plaster.run.sim_v2.fast import sim_v2_fast
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.run.sim_v2 import sim_v2_params
from plaster.tools.log.log import debug
from plaster.tools.schema import check
from plaster.tools.utils import data


def _rand_lognormals(logs, sigma):
    """Mock-point"""
    return np.random.lognormal(mean=logs, sigma=sigma, size=logs.shape)


def _gen_flus(sim_v2_params, pep_seq_df):
    flus = []
    pi_brights = []

    labelled_pep_df = pep_seq_df.join(
        sim_v2_params.df.set_index("amino_acid"), on="aa", how="left"
    )

    # p_bright = is the product of (1.0 - ) all the ways the dye can fail to be visible.
    labelled_pep_df["p_bright"] = (
        (1.0 - labelled_pep_df.p_failure_to_attach_to_dye)
        * (1.0 - labelled_pep_df.p_failure_to_bind_amino_acid)
        * (1.0 - labelled_pep_df.p_non_fluorescent)
    )

    for pep_i, group in labelled_pep_df.groupby("pep_i"):
        flu_float = group.ch_i.values
        flu = np.nan_to_num(flu_float, nan=sim_v2_fast.NO_LABEL).astype(
            sim_v2_fast.DyeType
        )
        flus += [flu]

        p_bright = np.nan_to_num(group.p_bright.values)
        pi_bright = np.zeros((len(flu),), dtype=sim_v2_fast.PIType)
        for i, p in enumerate(p_bright):
            pi_bright[i] = sim_v2_fast.prob_to_p_i(p)
        pi_brights += [pi_bright]

    return flus, pi_brights


def _dyemat_sim(sim_v2_params, flus, pi_brights, n_samples):
    """
    Run via the C fast_sim module a dyemat sim.

    Returns:
        dyemat: ndarray(n_uniq_dyetracks, n_channels, n_cycle)
        dyepep: ndarray(dye_i, pep_i, count)
        pep_recalls: ndarray(n_peps)
    """

    # TODO: bleach per channel
    dyemat, dyepeps, pep_recalls = sim_v2_fast.sim(
        flus,
        pi_brights,
        n_samples,
        sim_v2_params.n_channels,
        sim_v2_params.cycles_array(),
        sim_v2_params.error_model.dyes[0].p_bleach_per_cycle,
        sim_v2_params.error_model.p_detach,
        sim_v2_params.error_model.p_edman_failure,
        n_threads=1,  # TODO, tune
        rng_seed=sim_v2_params.random_seed,
    )

    return dyemat, dyepeps, pep_recalls


def _sample_pep_dyemat(dyepep_group, n_samples_per_pep):
    """
    Sample a peptide's dyetracks with probability weighted
    by the dyepep's counts.

    Arguments:
        dyepep_group: ndarray[n_dts_for_one_peptide, 3]
        n_samples_per_pep: number of samples requested
    Returns:
        Array of indicies into the dyetracks (ie sampling 0-th column
        of dyetracks)
    """
    counts = dyepep_group[:, 2].astype(np.float32)
    sum_counts = counts.sum()
    if dyepep_group.shape[0] > 0 and sum_counts > 0:
        prob = counts / counts.sum()
        return np.random.choice(dyepep_group[:, 0], n_samples_per_pep, p=prob)
    else:
        return np.zeros((0,), dtype=int)


def _radmat_from_sampled_pep_dyemat(
    dyemat, ch_params, n_channels, output_radmat, pep_i
):
    if dyemat.shape[0] > 0:
        for ch_i in range(n_channels):
            ch_log_beta = math.log(ch_params[ch_i].beta)
            ch_sigma = ch_params[ch_i].sigma

            # dyemat can have zeros, nan these to prevent log(0)
            # Also, the dyemat is int so this needs to be converted to float
            dm_nan = dyemat[:, ch_i, :].astype(float)
            dm_nan[dm_nan == 0] = np.nan
            logs = np.log(dm_nan)  # log(nan) == nan

            # Remember: log(a) + log(b) == log(a*b)
            # So we're scaling the dyes by beta and taking the log
            ch_radiometry = _rand_lognormals(ch_log_beta + logs, ch_sigma)

            output_radmat[pep_i, :, ch_i, :] = np.nan_to_num(ch_radiometry)


def _radmat_sim(dyemat, dyepeps, ch_params, n_samples_per_pep, n_channels, n_cycles):
    """
    Generate a radmat with equal number of samples per peptide.

    Each peptide has an unknown number of dyetracks that it can
    generate. These are in the dyepeps arrays.

    Sort the dyepeps by pep_i (column [1]) and then group them.

    Allocate an output array that is n_samples_per_pep for each peptide
    so that evey peptide gets an equal number of samples.

    The inner loop is implemented in Cython, so each pep_i group
    is passed along to Cython where it will fill in the samples.

    """

    # SORT dyepeps by peptide (col 1) first then by count (col 2)
    # Note that np.lexsort puts the primary sort key LAST in the argument
    sorted_dyepeps = dyepeps[np.lexsort((-dyepeps[:, 2], dyepeps[:, 1]))]

    # GROUP sorted_dyepeps by peptide using trick described here:
    # https://stackoverflow.com/a/43094244
    # This results in a list of numpy arrays.
    # Note there might be holes, unlikely but possible that
    # not every peptide has an entry in grouped_dyepep_rows therefore
    # this can not be treated as a lookup table by pep_i)
    grouped_dyepep_rows = np.split(
        sorted_dyepeps,
        np.cumsum(np.unique(sorted_dyepeps[:, 1], return_counts=True)[1])[:-1],
    )

    if len(dyepeps[:, 1]) > 0:
        n_peps = int(np.max(dyepeps[:, 1]) + 1)
    else:
        n_peps = 0

    # TODO: Convert to ArrayResult
    output_radmat = np.zeros(
        (n_peps, n_samples_per_pep, n_channels, n_cycles), dtype=np.float32
    )
    output_true_dye_iz = np.zeros((n_peps, n_samples_per_pep), dtype=int)

    for group_i, dyepep_group in enumerate(grouped_dyepep_rows):
        # All of the pep_iz (column 1) should be the same since that's what a "group" is.
        if dyepep_group.shape[0] > 0:
            pep_i = dyepep_group[0, 1]
            sampled_dye_iz = _sample_pep_dyemat(dyepep_group, n_samples_per_pep)
            if sampled_dye_iz.shape[0] > 0:
                output_true_dye_iz[pep_i, :] = sampled_dye_iz
            _radmat_from_sampled_pep_dyemat(
                dyemat[sampled_dye_iz], ch_params, n_channels, output_radmat, pep_i
            )

    output_radmat = output_radmat.reshape(
        (n_peps * n_samples_per_pep, n_channels, n_cycles)
    )
    output_true_pep_iz = np.repeat(np.arange(n_peps), n_samples_per_pep)
    output_true_dye_iz = output_true_dye_iz.reshape((n_peps * n_samples_per_pep,))
    return output_radmat, output_true_pep_iz, output_true_dye_iz


def _any_identical_non_zero_rows(a, b):
    """
    Checks if two mats a and b are identical in ANY non-zero rows.
    """
    check.array_t(a, ndim=2)
    check.array_t(b, ndim=2)

    arg_sample = data.arg_subsample(a, 100)
    a = a[arg_sample]
    b = b[arg_sample]

    zero_rows = np.all(a == 0, axis=1)
    a = a[~zero_rows]
    b = b[~zero_rows]

    if a.shape[0] > 0:
        return np.any(a == b)


def sim_v2(sim_v2_params, prep_result, progress=None, pipeline=None):
    train_flus = None
    train_dyemat = None
    train_dyepeps = None
    train_pep_recalls = None
    train_radmat = None
    train_true_pep_iz = None
    train_true_dye_iz = None
    test_flus = None
    test_dyemat = None
    test_radmat = None
    test_dyepeps = None
    test_pep_recalls = None
    test_true_pep_iz = None
    test_true_dye_iz = None

    # Training data
    #   * always includes decoys
    #   * may include radiometry
    # -----------------------------------------------------------------------
    train_flus, train_pi_brights = _gen_flus(sim_v2_params, prep_result.pepseqs())

    train_dyemat, train_dyepeps, train_pep_recalls = _dyemat_sim(
        sim_v2_params, train_flus, train_pi_brights, sim_v2_params.n_samples_train
    )

    # SORT dyepeps by dyetrack (col 0) first then reverse by count (col 2)
    # Note that np.lexsort puts the primary sort key LAST in the argument
    train_dyepeps = train_dyepeps[
        np.lexsort((-train_dyepeps[:, 2], train_dyepeps[:, 0]))
    ]

    if sim_v2_params.train_includes_radmat:
        train_radmat, train_true_pep_iz, train_true_dye_iz = _radmat_sim(
            train_dyemat.reshape(
                (
                    train_dyemat.shape[0],
                    sim_v2_params.n_channels,
                    sim_v2_params.n_cycles,
                )
            ),
            train_dyepeps,
            sim_v2_params.by_channel,
            sim_v2_params.n_samples_train,
            sim_v2_params.n_channels,
            sim_v2_params.n_cycles,
        )
    else:
        train_radmat, train_true_pep_iz, train_true_dye_iz = None, None, None

    # Test data
    #   * does not include decoys
    #   * always includes radiometry
    #   * may include dyetracks
    #   * skipped if is_survey
    # -----------------------------------------------------------------------
    if not sim_v2_params.is_survey:
        test_flus, test_pi_brights = _gen_flus(
            sim_v2_params, prep_result.pepseqs__no_decoys()
        )
        test_dyemat, test_dyepeps, test_pep_recalls = _dyemat_sim(
            sim_v2_params, test_flus, test_pi_brights, sim_v2_params.n_samples_test
        )

        # SORT dyepeps by dyetrack (col 0) first then reverse by count (col 2)
        # Note that np.lexsort puts the primary sort key LAST in the argument
        test_dyepeps = test_dyepeps[
            np.lexsort((-test_dyepeps[:, 2], test_dyepeps[:, 0]))
        ]

        test_radmat, test_true_pep_iz, test_true_dye_iz = _radmat_sim(
            test_dyemat.reshape(
                (test_dyemat.shape[0], sim_v2_params.n_channels, sim_v2_params.n_cycles)
            ),
            test_dyepeps,
            sim_v2_params.by_channel,
            sim_v2_params.n_samples_test,
            sim_v2_params.n_channels,
            sim_v2_params.n_cycles,
        )

        if not sim_v2_params.allow_train_test_to_be_identical:
            # TASK: Add a dyepeps check
            # train_dyepeps_df = pd.DataFrame(train_dyepeps, columns=["dye_i", "pep_i", "count"])
            # test_dyepeps_df = pd.DataFrame(test_dyepeps, columns=["dye_i", "pep_i", "count"])
            # joined_df = train_dyepeps_df.set_index("pep_i").join(
            #     test_dyepeps_df.set_index("pep_i")
            # )

            if train_radmat is not None:
                check.affirm(
                    not _any_identical_non_zero_rows(train_radmat, test_radmat),
                    "Train and test sets are identical. Probably RNG bug.",
                )

        # if not sim_v2_params.test_includes_dyemat:
        #     test_dyemat, test_dyepeps_df = None, None

    # REMOVE all-zero rows (EXECPT THE FIRST which is the nul row)
    non_zero_rows = np.argwhere(test_true_pep_iz != 0).flatten()
    debug(non_zero_rows.shape, test_radmat.shape)
    test_radmat = test_radmat[non_zero_rows]
    debug(test_radmat.shape)
    test_true_pep_iz = test_true_pep_iz[non_zero_rows]
    test_true_dye_iz = test_true_dye_iz[non_zero_rows]

    """
    assert np.all(train_dyemat[0, :] == 0)
    debug(train_dyemat.shape)
    debug(np.all(train_dyemat[:, :] == 0, axis=1).sum())
    some_non_zero_row_args = np.argwhere(~np.all(train_dyemat[:, :] == 0, axis=1)).flatten()
    debug(some_non_zero_row_args.shape)
    some_non_zero_row_args = np.concatenate(([0], some_non_zero_row_args))
    train_dyemat = train_dyemat[some_non_zero_row_args]
    if train_radmat is not None:
        train_radmat = train_radmat[some_non_zero_row_args]
    if train_true_pep_iz is not None:
        train_true_pep_iz = train_true_pep_iz[some_non_zero_row_args]
    if train_true_dye_iz is not None:
        train_true_dye_iz = train_true_dye_iz[some_non_zero_row_args]

    if test_dyemat is not None:
        assert np.all(test_dyemat[0, :] == 0)
    some_non_zero_row_args = np.argwhere(~np.all(test_dyemat[:, :] == 0, axis=1)).flatten()
    some_non_zero_row_args = np.concatenate(([0], some_non_zero_row_args))
    test_dyemat = test_dyemat[some_non_zero_row_args]
    test_radmat = test_radmat[some_non_zero_row_args]
    test_true_pep_iz = test_true_pep_iz[some_non_zero_row_args]
    test_true_dye_iz = test_true_dye_iz[some_non_zero_row_args]
    """

    return SimV2Result(
        params=sim_v2_params,
        train_dyemat=train_dyemat,
        train_radmat=train_radmat,
        train_pep_recalls=train_pep_recalls,
        train_flus=train_flus,
        train_true_pep_iz=train_true_pep_iz,
        train_true_dye_iz=train_true_dye_iz,
        train_dyepeps=train_dyepeps,
        test_dyemat=test_dyemat,
        test_radmat=test_radmat,
        test_true_pep_iz=test_true_pep_iz,
        test_true_dye_iz=test_true_dye_iz,
    )
