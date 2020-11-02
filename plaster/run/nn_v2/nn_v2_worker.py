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
    def _run(radmat):
        with c_nn_v2.context(
            radmat=radmat,
            train_dyemat=sim_v2_result.flat_train_dyemat(),
            train_dyepeps=sim_v2_result.train_dyepeps,
            gain_model=sim_v2_result.params.to_error_model().to_gain_model(),
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
    n_phases = 3
    if sigproc_result is not None:
        n_phases += 1

    # RUN NN on test set
    # -----------------------------------------------------------------------
    test_radmat = sim_v2_result.flat_test_radmat()

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

    # RUN NN on sigproc_result if available
    # -----------------------------------------------------------------------
    sigproc_context = None
    if sigproc_result is not None:
        sigproc_radmat = sigproc_result.flat_radmat().astype(RadType)

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        sigproc_context = _run(sigproc_radmat)

    return NNV2Result(
        params=nn_v2_params,
        test_pred_pep_iz=test_context.pred_pep_iz,
        test_pred_dyt_iz=test_context.pred_dyt_iz,
        test_scores=test_context.pred_scores,
        test_true_pep_iz=sim_v2_result.test_true_pep_iz,
        test_peps_pr=test_peps_pr,
        test_peps_pr_abund=test_peps_pr_abund,
        sigproc_pred_pep_iz=None
        if sigproc_context is None
        else sigproc_context.pred_pep_iz,
        sigproc_scores=None if sigproc_context is None else sigproc_context.pred_scores,
        sigproc_pred_dyt_i=None
        if sigproc_context is None
        else sigproc_context.pred_dyt_iz,
    )


"""
def nn_v2(
    nn_v2_params,
    prep_result,
    sim_v2_result,
    sigproc_result,
    progress=None,
    pipeline=None,
):
    phase_i = 0
    n_phases = 3
    if sigproc_result is not None:
        n_phases += 1

    n_cycles = sim_v2_result.params.n_cycles
    n_channels = sim_v2_result.params.n_channels
    radmat_normalization = np.zeros((n_channels * n_cycles), np.float32)
    for ch in range(n_channels):
        for cy in range(n_cycles):
            radmat_normalization[ch * n_cycles + cy] = sim_v2_result.params.by_channel[
                ch
            ].beta

    assert np.all(radmat_normalization > 0.0)

    # RUN NN on test set
    # -----------------------------------------------------------------------
    unit_radmat = sim_v2_result.flat_test_radmat() / radmat_normalization

    if pipeline:
        pipeline.set_phase(phase_i, n_phases)
        phase_i += 1

    test_pred_pep_iz, test_scores, test_pred_dyt_iz = nn_v2_fast_2.fast_nn(
        unit_radmat,
        sim_v2_result.flat_train_dyemat(),
        sim_v2_result.train_dyepeps,
        n_neighbors=nn_v2_params.n_neighbors,
        n_threads=zap.get_cpu_limit(),
        progress=progress,
        run_against_all_dyetracks=nn_v2_params.run_against_all_dyetracks,
    )

    call_bag = CallBag(
        true_pep_iz=sim_v2_result.test_true_pep_iz,
        pred_pep_iz=test_pred_pep_iz,
        scores=test_scores,
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

    # RUN NN on sigproc_result if available
    # -----------------------------------------------------------------------
    if sigproc_result is not None:
        sigproc_unit_radmat = (
            sigproc_result.flat_radmat().astype(RadType) / radmat_normalization
        )

        if pipeline:
            pipeline.set_phase(phase_i, n_phases)
            phase_i += 1

        sigproc_pred_pep_iz, sigproc_scores, sigproc_pred_dyt_iz = nn_v2_fast_2.fast_nn(
            sigproc_unit_radmat,
            sim_v2_result.flat_train_dyemat(),
            sim_v2_result.train_dyepeps,
            n_neighbors=nn_v2_params.n_neighbors,
            n_threads=zap.get_cpu_limit(),
            progress=progress,
            run_against_all_dyetracks=nn_v2_params.run_against_all_dyetracks,
        )
    else:
        sigproc_pred_pep_iz = None
        sigproc_scores = None
        sigproc_pred_dyt_iz = None

    return NNV2Result(
        params=nn_v2_params,
        test_pred_pep_iz=test_pred_pep_iz,
        test_pred_dyt_iz=test_pred_dyt_iz,
        test_scores=test_scores,
        test_true_pep_iz=sim_v2_result.test_true_pep_iz,
        test_peps_pr=test_peps_pr,
        test_peps_pr_abund=test_peps_pr_abund,
        sigproc_pred_pep_iz=sigproc_pred_pep_iz,
        sigproc_scores=sigproc_scores,
        sigproc_pred_dyt_i=sigproc_pred_dyt_iz,
    )
"""
