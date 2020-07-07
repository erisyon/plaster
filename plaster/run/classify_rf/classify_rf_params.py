from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class ClassifyRFParams(Params):
    defaults = Munch()

    schema = s(s.is_kws_r())
