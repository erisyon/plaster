from munch import Munch
from plaster.tools.schema.schema import Params
from plaster.tools.schema.schema import Schema as s
from plaster.run.error_model import GainModel, CycleBalance


class NNV2Params(Params):
    defaults = Munch(
        include_training_set=False,
        n_neighbors=8,
        dt_score_bias=0.1,
        include_sigproc=False,
        run_against_all_dyetracks=False,
        run_row_k_fit=False,
        dyetrack_n_counts=None,
        dyetrack_n_cycles=None,
        row_k_score_factor=0.0,
    )

    schema = s(
        s.is_kws_r(
            include_training_set=s.is_bool(),
            n_neighbors=s.is_int(),
            dt_score_bias=s.is_float(),
            include_sigproc=s.is_bool(),
            run_row_k_fit=s.is_bool(),
            run_against_all_dyetracks=s.is_bool(),
            gain_model=s.is_type(GainModel),
            dyetrack_n_counts=s.is_int(noneable=True),
            dyetrack_n_cycles=s.is_int(noneable=True),
            row_k_score_factor=s.is_float(),
            cycle_balance=s.is_type(CycleBalance, noneable=True),
        )
    )
