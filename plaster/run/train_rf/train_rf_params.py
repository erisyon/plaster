from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class TrainRFParams(Params):
    defaults = Munch(
        n_subsample=1_000,
        n_estimators=10,
        min_samples_leaf=50,
        max_depth=None,
        max_features="auto",
        max_leaf_nodes=None,
    )

    schema = s(
        s.is_kws_r(
            n_subsample=s.is_int(),
            n_estimators=s.is_int(),
            min_samples_leaf=s.is_int(),
            max_depth=s.is_int(noneable=True),
            max_features=s.is_type(object),
            max_leaf_nodes=s.is_int(noneable=True),
        )
    )
