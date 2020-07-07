from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class TestRFParams(Params):
    defaults = Munch(include_training_set=False, keep_all_class_scores=False)

    schema = s(
        s.is_kws_r(include_training_set=s.is_bool(), keep_all_class_scores=s.is_bool(),)
    )
