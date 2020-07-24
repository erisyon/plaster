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
from plaster.run.sim_v2.fast import sim_v2_fast
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.run.sim_v2 import sim_v2_params
from plaster.tools.log.log import debug


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

    # TODO: bleach each channel
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


def _radmat_sim(dyemat, dyepeps, ch_params):
    """
    TODO: This can be sped up in a variety of ways.
    For one, we could avoid the entire call and move
    this into a generator-like function in C instead
    of realizing the entire block of memory.

    That said, this large file simulats what the data would
    look like coming from the scope so it is somewhat
    nice that the simulator goes through the same code path.

    But also, this just needs to be moved into C.
    """

    if dyepeps.shape[0] == 0:
        n_peps = 0
    else:
        n_peps = int(np.max(dyepeps[:, 1]) + 1)

    n_channels, n_cycles = dyemat.shape[-2:]

    n_samples_total = np.sum(dyepeps[:, 2])

    radiometry = np.zeros(
        (n_samples_total, n_channels, n_cycles), dtype=sim_v2_params.RadType
    )
    true_pep_iz = np.zeros((n_samples_total), dtype=int)

    sample_i = 0
    for pep_i in range(n_peps):
        _dyepep_rows = dyepeps[dyepeps[:, 1] == pep_i]

        for row in _dyepep_rows:
            dt_i, _, count = row

            if count > 0:
                for ch in range(n_channels):
                    log_ch_beta = math.log(ch_params[ch].beta)
                    ch_sigma = ch_params[ch].sigma

                    # dyemat can have zeros, nan these to prevent log(0)
                    dm_nan = dyemat[dt_i, ch, :].astype(float)
                    dm_nan[dm_nan == 0] = np.nan

                    dm_nan = np.repeat(dm_nan[None, :], count, axis=0)

                    logs = np.log(dm_nan)  # log(nan) == nan

                    # Remember: log(a) + log(b) == log(a*b)
                    # So we're scaling the dyes by beta and taking the log
                    ch_radiometry = _rand_lognormals(log_ch_beta + logs, ch_sigma)

                    radiometry[sample_i : int(sample_i + count), ch, :] = np.nan_to_num(
                        ch_radiometry
                    )

                true_pep_iz[sample_i : int(sample_i + count)] = pep_i

            sample_i = int(sample_i + count)

    assert sample_i == n_samples_total

    return radiometry.reshape((radiometry.shape[0], radiometry.shape[1] * radiometry.shape[2])), true_pep_iz


def sim_v2(sim_v2_params, prep_result, progress=None, pipeline=None):
    train_flus = None
    train_dyemat = None
    train_dyepeps = None
    train_pep_recalls = None
    train_radmat = None
    train_true_pep_iz = None
    test_flus = None
    test_dyemat = None
    test_dyepeps = None
    test_pep_recalls = None

    # Training data
    #   * always includes decoys
    #   * may include radiometry
    # -----------------------------------------------------------------------
    train_flus, train_pi_brights = _gen_flus(sim_v2_params, prep_result.pepseqs())

    train_dyemat, train_dyepeps, train_pep_recalls = _dyemat_sim(
        sim_v2_params, train_flus, train_pi_brights, sim_v2_params.n_samples_train
    )

    # if sim_v2_params.train_includes_radmat:
    #     train_radmat,  = _radmat_sim(
    #         train_dyemat, train_dyepeps
    #     )

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
        test_radmat, test_true_pep_iz = _radmat_sim(
            test_dyemat.reshape(
                (test_dyemat.shape[0], sim_v2_params.n_channels, sim_v2_params.n_cycles)
            ),
            test_dyepeps,
            sim_v2_params.by_channel,
        )

        if not sim_v2_params.test_includes_dyemat:
            test_dyemat, test_dyepeps_df = None, None

    return SimV2Result(
        params=sim_v2_params,
        train_dyemat=train_dyemat,
        train_pep_recalls=train_pep_recalls,
        train_flus=train_flus,
        train_dyepeps=train_dyepeps,
        test_dyemat=test_dyemat,
        test_radmat=test_radmat,
        test_true_pep_iz=test_true_pep_iz,
    )
