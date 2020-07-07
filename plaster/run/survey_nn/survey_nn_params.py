from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class SurveyNNParams(Params):
    # Maybe I'll inherit params from TestNN, but let's see what we want first...

    defaults = Munch()
    schema = s(s.is_kws_r())
