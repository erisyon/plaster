import numpy as np
import pandas as pd
from plaster.run.call_bag import CallBag
from plaster.run.nn_v2.c import nn_v2 as c_nn_v2
from plaster.run.nn_v2.nn_v2_result import NNV2Result
from plaster.run.sim_v2.sim_v2_result import DyeType, RadType
from plaster.tools.log.log import debug
from plaster.tools.zap import zap
from plaster.tools.schema import check


def triangle_dyemat(n_cycles, n_dyes):
    """
    Generate a "triangle" dyemat.
    Example: n_cycles = 3, n_dyes = 2
        0 0 0
        1 0 0
        1 1 0
        1 1 1
        2 0 0
        2 1 0
        2 1 1
        2 2 0
        2 2 1
        2 2 2
    """
    dyemat = []
    for cy0 in range(n_cycles + 1):
        row = np.zeros((n_cycles,), dtype=DyeType)
        row[0:cy0] = 1

        dyemat += [row.copy()]

        if n_dyes == 1:
            continue

        for cy1 in range(cy0):
            row[0 : cy1 + 1] = 2
            dyemat += [row.copy()]

            if n_dyes == 2:
                continue

            for cy2 in range(cy1):
                row[0 : cy2 + 1] = 3
                dyemat += [row.copy()]

    dyemat = np.array(dyemat, dtype=DyeType)
    rev_cols = [dyemat[:, cy] for cy in range(dyemat.shape[1] - 1, -1, -1)]
    dyemat = dyemat[np.lexsort(rev_cols)]

    dyepeps = np.zeros((dyemat.shape[0] - 1, 3), dtype=np.uint64)
    dyepeps[:, 0] = np.arange(dyemat.shape[0] - 1) + 1  # Skip 0
    dyepeps[:, 1] = 1
    dyepeps[:, 2] = 1

    return dyemat, dyepeps


def nn_v2(
    nn_v2_params,
    prep_result,
    sim_v2_result,
    sigproc_result,
    progress=None,
    pipeline=None,
):
    from plaster.run.nn_v2.c.nn_v2 import init as nn_v2_c_init

    nn_v2_c_init()

    if sim_v2_result is not None:
        n_cols = sim_v2_result.flat_train_dyemat().shape[1]
    else:
        n_cols = sigproc_result.n_cols

    def _run(radmat, dyemat, dyepeps):
        with c_nn_v2.context(
            radmat=radmat,
            train_dyemat=dyemat,
            train_dyepeps=dyepeps,
            gain_model=nn_v2_params.gain_model,
            n_neighbors=nn_v2_params.n_neighbors,
            run_row_k_fit=nn_v2_params.run_row_k_fit,
            run_against_all_dyetracks=nn_v2_params.run_against_all_dyetracks,
            row_k_score_factor=nn_v2_params.row_k_score_factor,
        ) as nn_v2_context:
            # _nn_v2.c chokes if a batch is larger than 1024*16
            batches = zap.make_batch_slices(
                n_rows=radmat.shape[0], _batch_size=1024 * 16
            )
            work_orders = [
                dict(
                    fn=c_nn_v2.do_classify_radrows,
                    radrow_start_i=batch[0],
                    n_radrows=batch[1] - batch[0],
                    nn_v2_context=nn_v2_context,
                )
                for batch in batches
            ]
            zap.work_orders(
                work_orders,
                _process_mode=False,
                _trap_exceptions=False,
                _progress=progress,
            )
            return nn_v2_context

    phase_i = 0
    n_phases = 5

    # RUN NN on test set if requested
    # -----------------------------------------------------------------------
    test_df = None
    test_peps_pr = None
    test_peps_pr_abund = None
    if sim_v2_result is not None:
        test_radmat = sim_v2_result.flat_test_radmat().astype(RadType)

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        test_context = _run(
            test_radmat,
            dyemat=sim_v2_result.flat_train_dyemat(),
            dyepeps=sim_v2_result.train_dyepeps,
        )

        test_df = test_context.to_dataframe()
        test_df["true_pep_iz"] = sim_v2_result.test_true_pep_iz
        test_df["true_dyt_iz"] = sim_v2_result.test_true_dye_iz

        call_bag = CallBag(
            true_pep_iz=sim_v2_result.test_true_pep_iz,
            pred_pep_iz=test_context.pred_pep_iz,
            scores=test_context.pred_scores,
            prep_result=prep_result,
            sim_result=sim_v2_result,
        )

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        test_peps_pr = call_bag.pr_curve_by_pep(progress=progress)

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        test_peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance(progress=progress)

    # RUN NN on train set if requested
    # -----------------------------------------------------------------------
    train_df = None
    if nn_v2_params.include_training_set:
        train_radmat = sim_v2_result.flat_tr_radmat().astype(RadType)

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        train_context = _run(
            train_radmat,
            dyemat=sim_v2_result.flat_train_dyemat(),
            dyepeps=sim_v2_result.train_dyepeps,
        )

        train_df = train_context.to_dataframe()
        train_df["true_pep_iz"] = sim_v2_result.train_true_pep_iz
        train_df["true_dyt_iz"] = sim_v2_result.train_true_dye_iz

    # RUN NN on sigproc_result if requested
    # -----------------------------------------------------------------------
    dyemat = None
    dyepeps = None
    sigproc_df = None
    if sigproc_result is not None:
        sigproc_radmat = sigproc_result.sig(flat_chcy=True).astype(RadType)

        if nn_v2_params.n_rows_limit is not None:
            sigproc_radmat = sigproc_radmat[0 : nn_v2_params.n_rows_limit]

        if nn_v2_params.cycle_balance is not None:
            check.list_t(nn_v2_params.cycle_balance.balance, float)
            assert len(nn_v2_params.cycle_balance.balance) == sigproc_result.n_cycles
            sigproc_radmat = sigproc_radmat * np.repeat(
                np.array(nn_v2_params.cycle_balance.balance), sigproc_result.n_channels
            ).astype(sigproc_radmat.dtype)

        if sigproc_radmat.shape[1] != n_cols:
            raise TypeError(
                f"In nn_v2 sigproc_radmat did not have same number of columns as training dyemat {sigproc_radmat.shape[1]} vs {n_cols}"
            )

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        if nn_v2_params.dyetrack_n_cycles is not None:
            assert nn_v2_params.dyetrack_n_counts is not None
            assert (
                nn_v2_params.dyetrack_n_counts < 4
            )  # Defend against crazy large memory alloc

            dyemat, dyepeps = triangle_dyemat(
                nn_v2_params.dyetrack_n_cycles, nn_v2_params.dyetrack_n_counts
            )
            sigproc_context = _run(sigproc_radmat, dyemat, dyepeps)

        else:
            sigproc_context = _run(
                sigproc_radmat,
                dyemat=sim_v2_result.flat_train_dyemat(),
                dyepeps=sim_v2_result.train_dyepeps,
            )

        sigproc_df = sigproc_context.to_dataframe()

    return NNV2Result(
        params=nn_v2_params,
        _test_calls=test_df,
        _train_calls=train_df,
        _sigproc_calls=sigproc_df,
        _test_peps_pr=test_peps_pr,
        _test_peps_pr_abund=test_peps_pr_abund,
        _test_all=None,  # TODO
        _train_all=None,  # TODO
        _sigproc_all=None,  # TODO
        _dyemat=dyemat,
        _dyepeps=dyepeps,
    )
