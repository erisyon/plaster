from plaster.run.sim_v1.sim_v1_params import SimV1Params
from plaster.run.sim_v1 import sim_v1_worker
from plaster.run.error_model import ErrorModel


def result_from_prep_fixture(
    prep_result, n_labels=1, n_edmans=5, error_model=ErrorModel.from_defaults(1)
):

    sim_v1_params = SimV1Params.construct_from_aa_list(
        [chr(ord("A") + i) for i in range(n_labels)],
        error_model=error_model,
        n_pres=1,
        n_mocks=0,
        n_edmans=n_edmans,
    )

    return sim_v1_worker.sim_v1(sim_v1_params, prep_result)
