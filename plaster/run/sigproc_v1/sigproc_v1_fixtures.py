from plaster.run.sigproc_v1.sigproc_v1_params import SigprocV1Params
from plaster.run.sigproc_v1.sigproc_v1_result import SigprocV1Result
from plaster.tools.log.log import debug


class SigprocV1ResultFixture(SigprocV1Result):
    def _load_field_prop(self, field_i, prop):
        debug(field_i, prop)


def simple_sigproc_result_fixture():
    params = SigprocV1Params()
    return SigprocV1ResultFixture(
        params=params, n_input_channels=1, n_channels=1, n_cycles=4,
    )
