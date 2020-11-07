import pandas as pd
import numpy as np
from plaster.run.call_bag import CallBag
from plaster.run.nn_v2.c import nn_v2 as c_nn_v2
from plaster.run.nn_v2.nn_v2_result import NNV2Result
from plaster.run.sim_v2.sim_v2_result import RadType
from plaster.tools.zap import zap
from plaster.tools.log.log import debug


def nn_v2(
    nn_v2_params,
    prep_result,
    sim_v2_result,
    sigproc_result,
    progress=None,
    pipeline=None,
):
    n_cols = sim_v2_result.flat_train_dyemat().shape[1]

    def _run(radmat):
        with c_nn_v2.context(
            radmat=radmat,
            train_dyemat=sim_v2_result.flat_train_dyemat(),
            train_dyepeps=sim_v2_result.train_dyepeps,
            gain_model=nn_v2_params.gain_model,
            n_neighbors=nn_v2_params.n_neighbors,
            run_row_k_fit=nn_v2_params.run_row_k_fit,
            run_against_all_dyetracks=nn_v2_params.run_against_all_dyetracks,
        ) as nn_v2_context:
            batches = zap.make_batch_slices(n_rows=radmat.shape[0])
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

    # RUN NN on test set
    # -----------------------------------------------------------------------
    test_radmat = sim_v2_result.flat_test_radmat().astype(RadType)

    if pipeline:
        pipeline.set_phase(phase_i, n_phases)
        phase_i += 1

    test_context = _run(test_radmat)

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

        train_context = _run(train_radmat)

        train_df = train_context.to_dataframe()
        train_df["true_pep_iz"] = sim_v2_result.train_true_pep_iz
        train_df["true_dyt_iz"] = sim_v2_result.train_true_dye_iz

    # RUN NN on sigproc_result if available
    # -----------------------------------------------------------------------
    sigproc_context = None
    if sigproc_result is not None:
        sigproc_radmat = sigproc_result.sig(flat_chcy=True).astype(RadType)

        if sigproc_radmat.shape[1] != n_cols:
            raise TypeError(f"In nn_v2 sigproc_radmat did not have same number of columns as training dyemat {sigproc_radmat.shape[1]} vs {n_cols}")

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        sigproc_context = _run(sigproc_radmat)

    test_df = test_context.to_dataframe()
    test_df["true_pep_iz"] = sim_v2_result.test_true_pep_iz
    test_df["true_dyt_iz"] = sim_v2_result.test_true_dye_iz

    return NNV2Result(
        params=nn_v2_params,
        _test_calls=test_df,
        _train_calls=train_df,
        _sigproc_calls=(
            None if sigproc_context is None else sigproc_context.to_dataframe()
        ),
        test_peps_pr=test_peps_pr,
        test_peps_pr_abund=test_peps_pr_abund,
        _test_all=None,  # TODO
        _train_all=None,  # TODO
        _sigproc_all=None,  # TODO
    )
