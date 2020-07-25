from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.error_model import ErrorModel


def result_from_prep_fixture(prep_result, n_labels=1, n_edmans=5, error_model=ErrorModel.from_defaults(1)):

    sim_v2_params = SimV2Params.construct_from_aa_list(
        [chr(ord("A") + i) for i in range(n_labels)],
        error_model=error_model,
        n_pres=1,
        n_mocks=0,
        n_edmans=n_edmans,
    )

    return sim_v2_worker.sim_v2(sim_v2_params, prep_result)

