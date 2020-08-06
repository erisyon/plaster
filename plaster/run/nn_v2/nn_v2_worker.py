import numpy as np
from plaster.run.nn_v2.nn_v2_result import NNV2Result
from plaster.run.nn_v2.fast import nn_v2_fast
from plaster.run.call_bag import CallBag
from plaster.tools.log.log import prof


def nn_v2(nn_v2_params, prep_result, sim_v2_result, sigproc_result):
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

    test_pred_pep_iz, test_scores, test_pred_dye_iz = nn_v2_fast.fast_nn(
        unit_radmat,
        sim_v2_result.flat_train_dyemat(),
        sim_v2_result.train_dyepeps,
        n_neighbors=nn_v2_params.n_neighbors,
        n_threads=4,
    )

    call_bag = CallBag(
        true_pep_iz=sim_v2_result.test_true_pep_iz,
        pred_pep_iz=test_pred_pep_iz,
        scores=test_scores,
        prep_result=prep_result,
        sim_result=sim_v2_result,
    )

    test_peps_pr = call_bag.pr_curve_by_pep()
    test_peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance()

    # RUN NN on sigproc_result if available
    # -----------------------------------------------------------------------
    if sigproc_result is not None:
        sigproc_unit_radmat = sigproc_result.flat_radmat() / radmat_normalization
        sigproc_pred_pep_iz, sigproc_scores, sigproc_pred_dye_iz = nn_v2_fast.fast_nn(
            sigproc_unit_radmat,
            sim_v2_result.flat_train_dyemat(),
            sim_v2_result.train_dyepeps,
            n_neighbors=nn_v2_params.n_neighbors,
            n_threads=4,
        )
    else:
        sigproc_pred_pep_iz = None
        sigproc_scores = None
        sigproc_pred_dye_iz = None

    return NNV2Result(
        params=nn_v2_params,
        test_pred_pep_iz=test_pred_pep_iz,
        test_pred_dye_iz=test_pred_dye_iz,
        test_scores=test_scores,
        test_true_pep_iz=sim_v2_result.test_true_pep_iz,
        test_peps_pr=test_peps_pr,
        test_peps_pr_abund=test_peps_pr_abund,
        sigproc_pred_pep_iz=sigproc_pred_pep_iz,
        sigproc_scores=sigproc_scores,
        sigproc_pred_dye_i=sigproc_pred_dye_iz,
    )
