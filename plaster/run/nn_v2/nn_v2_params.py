from munch import Munch
from plaster.tools.schema.schema import Params
from plaster.tools.schema.schema import Schema as s


class NNV2Params(Params):
    defaults = Munch(
        include_training_set=False,
        n_neighbors=8,
        dt_score_bias=0.1,
        include_sigproc=False,
    )

    schema = s(
        s.is_kws_r(
            include_training_set=s.is_bool(),
            n_neighbors=s.is_int(),
            dt_score_bias=s.is_float(),
            include_sigproc=s.is_bool(),
        )
    )
