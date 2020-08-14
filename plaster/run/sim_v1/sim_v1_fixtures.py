from plaster.run.sim_v1.sim_v1_params import SimV1Params
from plaster.run.sim_v1 import sim_v1_worker
from plaster.run.error_model import ErrorModel


def result_from_prep_fixture(
    prep_result, labels, n_edmans=5, error_model=None
):
    labels = labels.split(",")
    n_labels = len(labels)
    if error_model is None:
        error_model = ErrorModel.from_defaults(n_labels)

    sim_v1_params = SimV1Params.construct_from_aa_list(
        labels,
        error_model=error_model,
        n_pres=1,
        n_mocks=0,
        n_edmans=n_edmans,
    )

    return sim_v1_worker.sim_v1(sim_v1_params, prep_result)
