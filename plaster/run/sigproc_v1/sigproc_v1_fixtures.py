from plaster.run.sigproc_v1.sigproc_v1_params import SigprocV1Params
from plaster.run.sigproc_v1.sigproc_v1_result import SigprocV1Result
from plaster.run.sim_v2.sim_v2_worker import sim_v2
from plaster.tools.log.log import debug
from plaster.run.sim_v2.sim_v2_fixtures import result_from_prep_fixture

"""
class SigprocV1ResultFixture(SigprocV1Result):
    def _load_field_prop(self, field_i, prop):
        if prop == "signal_radmat":
            # self.prep_result.
            sim_v2_result = result_from_prep_fixture(self.prep_result)
            _radmat_from_sampled_pep_dyemat(
                sim_v2_result.train_dyemat,
                ch_params,
                n_channels,
                output_radmat,
                pep_i
            )

            sim_v2(sim_v2_params, self.prep_result)

        else:
            debug(prop)
            raise NotImplementedError

    @property
    def n_fields(self):
        return 1


def simple_sigproc_result_fixture(prep_result):
    params = SigprocV1Params()
    return SigprocV1ResultFixture(
        params=params, n_input_channels=1, n_channels=1, n_cycles=4, prep_result=prep_result
    )
"""
