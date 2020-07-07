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
from plaster.run.sim_v2 import sim_v2_fast
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.run.base_result import ArrayResult


def _gen_flus(sim_v2_params, pep_seq_df):
    flus = []
    labelled_pep_df = pep_seq_df.join(
        sim_v2_params.df.set_index("amino_acid"), on="aa", how="left"
    )
    for pep_i, group in labelled_pep_df.groupby("pep_i"):
        flu_float = group.ch_i.values
        flu = np.nan_to_num(flu_float, nan=sim_v2_fast.NO_LABEL).astype(
            sim_v2_fast.DyeType
        )
        flus += [flu]
    return flus


def _dyemat_sim(sim_v2_params, flus, n_samples):
    """
    Run via the C fast_sim module a dyemat sim.

    Outputs:
        dyemat: ndarray(n_uniq_dyetracks, n_channels, n_cycle)
        dyepep_df: Dataframe(dye_i, pep_i, count)
    """

    # TODO: bleach each channel
    # TODO: Include the p_bright calculations
    dyemat, dyepeps_df = sim_v2.sim(
        flus,
        n_samples,
        sim_v2_params.n_channels,
        sim_v2_params.cycles_array(),
        sim_v2_params.error_model.dyes[0].p_bleach_per_cycle,
        sim_v2_params.error_model.p_detach,
        sim_v2_params.error_model.p_edman_failure,
        n_threads=1,
        rng_seed=sim_v2_params.random_seed,
    )

    return dyemat, dyepeps_df


def _radmat_sim(sim_v2_params, n_samples, dyemat, dyepeps_df):
    raise NotImplementedError


def sim(sim_v2_params, prep_result, progress=None, pipeline=None):
    (
        train_flus,
        train_dyemat,
        train_dyepeps_df,
        train_radmat,
        train_radmat_true_pep_iz,
        test_flus,
        test_dyemat,
        test_dyepeps_df,
        test_radmat,
        test_radmat_true_pep_iz,
    ) = [None] * 10

    # Training data
    #   * always includes decoys
    #   * may include radiometry
    # -----------------------------------------------------------------------
    train_flus = _gen_flus(sim_v2_params, prep_result.pepseqs())
    train_dyemat, train_dyepeps_df = _dyemat_sim(
        sim_v2_params, train_flus, sim_v2_params.n_samples_train
    )

    if sim_v2_params.train_includes_radmat:
        train_radmat, train_radmat_true_pep_iz = _radmat_sim(
            train_dyemat, train_dyepeps_df
        )

    # Test data
    #   * does not include decoys
    #   * always includes radiometry
    #   * may include dyetracks
    #   * skipped if is_survey
    # -----------------------------------------------------------------------
    if not sim_v2_params.is_survey:
        test_flus = _gen_flus(sim_v2_params, prep_result.pepseqs__no_decoys())
        test_dyemat, test_dyepeps_df = _dyemat_sim(
            sim_v2_params, test_flus, sim_v2_params.n_samples_test
        )
        test_radmat, test_radmat_true_pep_iz = _radmat_sim(test_dyemat, test_dyepeps_df)

        if not sim_v2_params.test_includes_dyemat:
            test_dyemat, test_dyepeps_df = None, None

    return SimV2Result(
        params=sim_v2_params,
        train_flus=train_flus,
        train_dyemat=train_dyemat,
        train_dyepeps_df=train_dyepeps_df,
        train_radmat=train_radmat,
        train_radmat_true_pep_iz=train_radmat_true_pep_iz,
        test_flus=test_flus,
        test_dyemat=test_dyemat,
        test_dyepeps_df=test_dyepeps_df,
        test_radmat=test_radmat,
        test_radmat_true_pep_iz=test_radmat_true_pep_iz,
    )


"""



    if sim_v2_params.is_survey:
        test_dyemat = None
        test_radmat = None
        test_recalls = None
        test_flus = None
        test_flu_remainders = None
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
        ) = _run_sim(
            sim_v2_params,
            prep_result.pepseqs__no_decoys(),
            name="test",
            n_peps=n_peps,
            n_samples=sim_v2_params.n_samples_test,
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

        if (
                train_rows.shape[0] > 0
                and not sim_v2_params.allow_train_test_to_be_identical
        ):
            any_differences = np.any(np.diagonal(cdist(train_rows, test_rows)) != 0.0)
            check.affirm(any_differences, "Train and test sets are identical")

"""
