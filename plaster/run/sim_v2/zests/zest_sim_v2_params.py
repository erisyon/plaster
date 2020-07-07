from munch import Munch
from zest import zest
from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.run.error_model import ErrorModel
from plaster.tools.log.log import debug
from plaster.tools.utils import utils


def zest_sim_v2_params():
    def it_copies():
        error_model = ErrorModel.from_defaults(n_channels=2)
        src_params = SimV2Params.construct_from_aa_list(
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

    def it_makes_cycles_array():
        error_model = ErrorModel.from_defaults(n_channels=2)
        src_params = SimV2Params.construct_from_aa_list(
            ["DE", "Y"], error_model=error_model, n_pres=1, n_mocks=2, n_edmans=3
        )

        assert src_params.cycles_array().tolist() == [0, 1, 1, 2, 2, 2]

    zest()
