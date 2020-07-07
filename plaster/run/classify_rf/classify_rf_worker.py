from plaster.tools.utils import utils
from plaster.run.call_bag import CallBag
from plaster.run.classify_rf.classify_rf_result import ClassifyRFResult


def classify_rf(
    classify_rf_params, train_rf_result, sigproc_result, sim_params, progress=None,
):
    if sigproc_result.n_channels != sim_params.n_channels:
        raise ValueError(
            f"Sigproc n_channels ({sigproc_result.n_channels}) does not match "
            f"classifier n_channels ({sim_params.n_channels})"
        )

    if sigproc_result.n_cycles != sim_params.n_cycles:
        raise ValueError(
            f"Sigproc n_cycles ({sigproc_result.n_cycles}) does not match "
            f"classifier n_cycles ({sim_params.n_cycles})"
        )

    pred_pep_iz, scores, all_class_scores = train_rf_result.classifier.classify(
        utils.mat_flatter(sigproc_result.signal_radmat()), progress
    )

    return ClassifyRFResult(
        params=classify_rf_params,
        pred_pep_iz=pred_pep_iz,
        scores=scores,
        all_class_scores=all_class_scores,
        peps_pr=peps_pr,
    )
