"""
Test the Random Forest classifier.
"""

import numpy as np
import pandas as pd
from plaster.run.call_bag import CallBag
from plaster.run.test_rf.test_rf_result import TestRFResult
from plaster.tools.log.log import debug


def test_rf(
    test_rf_params,
    prep_result,
    sim_result,
    train_rf_result,
    progress=None,
    pipeline=None,
):
    n_phases = 6 if test_rf_params.include_training_set else 3
    classifier = train_rf_result.classifier

    if pipeline is not None:
        pipeline.set_phase(0, n_phases)

    test_pred_pep_iz, test_scores, test_all_class_scores = classifier.classify(
        sim_result.flat_test_radmat(), test_rf_params.keep_all_class_scores, progress
    )
    test_true_pep_iz = sim_result.test_true_pep_iz()

    # We do some PR calculation during the task so that this information is readily
    # available in results & notebooks don't need to recompute it (costly).
    # TODO: it is probably worth optimizing this by only doing PR for proteins of
    # interest if this has been specified for the run, since otherwise we'll be
    # computing full PR curves for every peptide in the background which is
    # probably not interesting.
    #
    call_bag = CallBag(
        true_pep_iz=test_true_pep_iz,
        pred_pep_iz=test_pred_pep_iz,
        scores=test_scores,
        all_class_scores=test_all_class_scores,
        prep_result=prep_result,
        sim_result=sim_result,
    )

    if pipeline is not None:
        pipeline.set_phase(1, n_phases)

    if pipeline is not None:
        pipeline.set_phase(2, n_phases)

    test_peps_pr = call_bag.pr_curve_by_pep(progress=progress)

    # If there is abundance information, compute the abundance-adjusted PR
    # This call returns None if there is no abundance info avail.
    test_peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance(progress=progress)

    if test_rf_params.include_training_set:
        # Permit testing for over-fitting by classifying on the train data

        if pipeline is not None:
            pipeline.set_phase(3, n_phases)

        real_pep_iz = prep_result.peps__no_decoys().pep_i.values

        keep_rows = np.isin(sim_result.train_true_pep_iz, real_pep_iz)

        train_true_pep_iz = sim_result.train_true_pep_iz[keep_rows]
        train_radmat = sim_result.train_radmat[keep_rows]

        train_pred_pep_iz, train_scores, train_all_class_scores = classifier.classify(
            train_radmat, test_rf_params.keep_all_class_scores, progress
        )

        call_bag = CallBag(
            true_pep_iz=train_true_pep_iz,
            pred_pep_iz=train_pred_pep_iz,
            scores=train_scores,
            all_class_scores=train_all_class_scores,
            prep_result=prep_result,
            sim_result=sim_result,
        )

        if pipeline is not None:
            pipeline.set_phase(4, n_phases)

        train_peps_pr = call_bag.pr_curve_by_pep(progress=progress)

        if pipeline is not None:
            pipeline.set_phase(5, n_phases)

        train_peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance(progress=progress)

    else:
        (
            train_pred_pep_iz,
            train_scores,
            train_all_class_scores,
            train_true_pep_iz,
            train_peps_pr,
            train_peps_pr_abund,
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    return TestRFResult(
        params=test_rf_params,
        test_true_pep_iz=test_true_pep_iz,
        test_pred_pep_iz=test_pred_pep_iz,
        test_scores=test_scores,
        test_all_class_scores=test_all_class_scores,
        test_peps_pr=test_peps_pr,
        test_peps_pr_abund=test_peps_pr_abund,
        train_true_pep_iz=train_true_pep_iz,
        train_pred_pep_iz=train_pred_pep_iz,
        train_scores=train_scores,
        train_all_class_scores=train_all_class_scores,
        train_peps_pr=train_peps_pr,
        train_peps_pr_abund=train_peps_pr_abund,
    )
