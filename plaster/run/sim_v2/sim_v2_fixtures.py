from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.error_model import ErrorModel
from plaster.tools.schema import check


def result_from_prep_fixture(prep_result, labels, n_edmans=5, error_model=None):
    # Common labels: "DE", "C", "Y", "K", "H"
    check.t(labels, str)
    labels = labels.split(",")
    n_labels = len(labels)

    if error_model is None:
        error_model = ErrorModel.from_defaults(n_labels)

    sim_v2_params = SimV2Params.construct_from_aa_list(
        labels, error_model=error_model, n_pres=1, n_mocks=0, n_edmans=n_edmans,
    )

    return sim_v2_worker.sim_v2(sim_v2_params, prep_result)
