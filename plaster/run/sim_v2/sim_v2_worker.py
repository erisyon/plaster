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
import numpy as np
from plaster.run.sim_v2.c import sim_v2 as sim_v2_fast
from plaster.run.sim_v2.sim_v2_params import RadType
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.tools.log.log import debug, prof
from plaster.tools.schema import check
from plaster.tools.utils import data
from plaster.tools.zap.zap import get_cpu_limit


def _rand_normals(mu, sigma):
    """Mock-point"""
    return np.random.normal(mu, sigma)


"""
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

    # labelled_pep_df[["pep_i", "pep_offset_in_pro", "ch_i", "p_bright"]].values

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
"""


def _dyemat_sim(sim_v2_params, pcbs, n_samples, progress=None):
    """
    Run via the C fast_sim module a dyemat sim.

    Returns:
        dyemat: ndarray(n_uniq_dyetracks, n_channels, n_cycle)
        dyepep: ndarray(dye_i, pep_i, count)
        pep_recalls: ndarray(n_peps)
    """
    sim_v2_fast.init()

    # TODO: bleach per channel
    dyemat, dyepeps, pep_recalls = sim_v2_fast.sim(
        pcbs,
        n_samples,
        sim_v2_params.n_channels,
        len(sim_v2_params.labels),
        sim_v2_params.cycles_array(),
        sim_v2_params.error_model.dyes[0].p_bleach_per_cycle,
        sim_v2_params.error_model.p_detach,
        sim_v2_params.error_model.p_edman_failure,
        n_threads=get_cpu_limit(),
        rng_seed=sim_v2_params.random_seed,
        progress=progress,
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
    """
    Generate into output_radmat for every channel for each pep.

    arguments:
        dyemat: The sampled dyemat. This function's job is to convert each
            row of this into a row in output_radmat
        ch_params
        n_channels
        output_radmat
        pep_i
    """

    if dyemat.shape[0] > 0:
        for ch_i in range(n_channels):
            ch_beta = ch_params[ch_i].beta
            ch_sigma = ch_params[ch_i].sigma
            ch_zero_beta = ch_params[ch_i].zero_beta
            ch_zero_sigma = ch_params[ch_i].zero_sigma

            # CONVERT dyemat to float and MASK zeros to NAN
            dm_nan = dyemat[:, ch_i, :].astype(float)
            dark_mask = dm_nan == 0

            dm_nan[dark_mask] = np.nan
            logs = np.log(dm_nan * ch_beta)  # Note: log(nan) == nan

            ch_radiometry = np.exp(_rand_normals(logs, ch_sigma))

            # FILL-IN the zero-counts with a different distribution
            # That is, the darks do not follow the same distribution
            # Note the LACK of the np.exp because we assume that the
            # darks do no follow a log-normal.
            # MAKE a new dyemat-like matrix full of the mean of the zero
            # counts and then mask out all the non-dark areas with nan
            # Then we merge the dark_radiometry and the ch_radiometry
            dm_zero_beta = np.full_like(logs, ch_zero_beta)
            dm_zero_beta[~dark_mask] = np.nan
            dark_radiometry = _rand_normals(dm_zero_beta, ch_zero_sigma)

            # PASTE in the dark value into the ch_radiometry
            ch_radiometry = np.where(dark_mask, dark_radiometry, ch_radiometry)

            # STORE
            output_radmat[pep_i, :, ch_i, :] = ch_radiometry


def _radmat_sim(
    dyemat, dyepeps, ch_params, n_samples_per_pep, n_channels, n_cycles, progress=None
):
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
        (n_peps, n_samples_per_pep, n_channels, n_cycles), dtype=RadType
    )
    output_true_dye_iz = np.zeros((n_peps, n_samples_per_pep), dtype=int)

    n_groups = len(grouped_dyepep_rows)
    for group_i, dyepep_group in enumerate(grouped_dyepep_rows):
        # All of the pep_iz (column 1) should be the same since that's what a "group" is.
        if progress is not None:
            progress(group_i, n_groups, 0)
        if dyepep_group.shape[0] > 0:
            pep_i = dyepep_group[0, 1]
            sampled_dye_iz = _sample_pep_dyemat(dyepep_group, n_samples_per_pep)
            if sampled_dye_iz.shape[0] > 0:
                output_true_dye_iz[pep_i, :] = sampled_dye_iz

            _radmat_from_sampled_pep_dyemat(
                dyemat[sampled_dye_iz], ch_params, n_channels, output_radmat, pep_i,
            )

    output_radmat = output_radmat.reshape(
        (n_peps * n_samples_per_pep, n_channels, n_cycles)
    )
    output_true_pep_iz = np.repeat(np.arange(n_peps), n_samples_per_pep)
    output_true_dye_iz = output_true_dye_iz.reshape((n_peps * n_samples_per_pep,))

    # REMOVE all zero-rows (those that point to the nul dyetrack)
    keep_good_tracks = np.argwhere(output_true_dye_iz != 0).flatten()
    output_radmat = output_radmat[keep_good_tracks]
    output_true_pep_iz = output_true_pep_iz[keep_good_tracks]
    output_true_dye_iz = output_true_dye_iz[keep_good_tracks]

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


def _radmat_add_per_row_variance(radmat, row_k_sigma):
    if row_k_sigma is None:
        return radmat, np.ones((radmat.shape[0],), dtype=RadType)
    true_ks = np.random.normal(1.0, row_k_sigma, size=(radmat.shape[0],)).astype(
        RadType
    )
    return radmat * true_ks[:, None, None], true_ks


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
    test_true_ks = None

    phase_i = 0
    n_phases = 1
    if sim_v2_params.train_includes_radmat:
        n_phases += 1
    if not sim_v2_params.is_survey:
        n_phases += 2

    # Training data
    #   * always includes decoys
    #   * may include radiometry
    # -----------------------------------------------------------------------
    # debug("gen flus")
    # train_flus, train_pi_brights = _gen_flus(sim_v2_params, prep_result.pepseqs())
    # debug("gen flus done")

    if pipeline:
        pipeline.set_phase(phase_i, n_phases)
        phase_i += 1

    pepseqs = prep_result.pepseqs__with_decoys()
    pcbs = sim_v2_params.pcbs(pepseqs)
    train_dyemat, train_dyepeps, train_pep_recalls = _dyemat_sim(
        sim_v2_params, pcbs, sim_v2_params.n_samples_train, progress,
    )

    # SORT dyepeps by dyetrack (col 0) first then reverse by count (col 2)
    # Note that np.lexsort puts the primary sort key LAST in the argument
    train_dyepeps = train_dyepeps[
        np.lexsort((-train_dyepeps[:, 2], train_dyepeps[:, 0]))
    ]

    if sim_v2_params.train_includes_radmat:
        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

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
            progress,
        )
        train_radmat, train_true_ks = _radmat_add_per_row_variance(
            train_radmat, sim_v2_params.row_k_sigma
        )
    else:
        train_radmat, train_true_pep_iz, train_true_dye_iz, train_true_ks = (
            None,
            None,
            None,
            None,
        )

    # Test data
    #   * does not include decoys
    #   * always includes radiometry
    #   * may include dyetracks
    #   * skipped if is_survey
    # -----------------------------------------------------------------------
    if not sim_v2_params.is_survey:
        # test_flus, test_pi_brights = _gen_flus(
        #     sim_v2_params, prep_result.pepseqs__no_decoys()
        # )

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        test_dyemat, test_dyepeps, test_pep_recalls = _dyemat_sim(
            sim_v2_params,
            sim_v2_params.pcbs(prep_result.pepseqs__no_decoys()),
            sim_v2_params.n_samples_test,
            progress,
        )

        # SORT dyepeps by dyetrack (col 0) first then reverse by count (col 2)
        # Note that np.lexsort puts the primary sort key LAST in the argument
        test_dyepeps = test_dyepeps[
            np.lexsort((-test_dyepeps[:, 2], test_dyepeps[:, 0]))
        ]

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        test_radmat, test_true_pep_iz, test_true_dye_iz = _radmat_sim(
            test_dyemat.reshape(
                (test_dyemat.shape[0], sim_v2_params.n_channels, sim_v2_params.n_cycles)
            ),
            test_dyepeps,
            sim_v2_params.by_channel,
            sim_v2_params.n_samples_test,
            sim_v2_params.n_channels,
            sim_v2_params.n_cycles,
            progress,
        )
        test_radmat, test_true_ks = _radmat_add_per_row_variance(
            test_radmat, sim_v2_params.error_model.row_k_sigma
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

        # REMOVE all-zero rows (EXCEPT THE FIRST which is the nul row)
        non_zero_rows = np.argwhere(test_true_pep_iz != 0).flatten()
        test_radmat = test_radmat[non_zero_rows]
        test_true_pep_iz = test_true_pep_iz[non_zero_rows]
        test_true_dye_iz = test_true_dye_iz[non_zero_rows]
        test_true_ks = test_true_ks[non_zero_rows]

    sim_result_v2 = SimV2Result(
        params=sim_v2_params,
        train_dyemat=train_dyemat,
        train_radmat=train_radmat,
        train_pep_recalls=train_pep_recalls,
        train_true_pep_iz=train_true_pep_iz,
        train_true_dye_iz=train_true_dye_iz,
        train_dyepeps=train_dyepeps,
        train_true_ks=train_true_ks,
        test_dyemat=test_dyemat,
        test_radmat=test_radmat,
        test_true_pep_iz=test_true_pep_iz,
        test_true_dye_iz=test_true_dye_iz,
        test_true_ks=test_true_ks,
        _flus=None,
    )

    if sim_v2_params.generate_flus:
        sim_result_v2._generate_flu_info(prep_result)

    return sim_result_v2
