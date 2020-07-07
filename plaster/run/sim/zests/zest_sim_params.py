from munch import Munch
from zest import zest
from plaster.run.sim.sim_params import SimParams
from plaster.run.error_model import ErrorModel


def zest_sim_params():
    def it_copies():
        error_model = ErrorModel.from_defaults(n_channels=2)
        src_params = SimParams.construct_from_aa_list(
            ["DE", "Y"], error_model=error_model
        )
        src_params._build_join_dfs()
        src_bleach = src_params.error_model.dyes[0].p_bleach_per_cycle
        dst_params = src_params.copy()
        dst_params.error_model.set_dye_param("p_bleach_per_cycle", 1.0)
        _src_bleach = src_params.error_model.dyes[0].p_bleach_per_cycle
        _dst_bleach = dst_params.error_model.dyes[0].p_bleach_per_cycle
        assert _src_bleach == src_bleach
        assert _dst_bleach == 1.0

    zest()
