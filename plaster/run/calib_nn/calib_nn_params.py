from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params
from plaster.tools.log.log import debug


class CalibNNParams(Params):
    defaults = Munch()

    schema = s(
        s.is_kws_r(
            mode=s.is_str(),
            n_pres=s.is_int(),
            n_mocks=s.is_int(),
            n_edmans=s.is_int(),
            dye_names=s.is_list(s.is_str()),
            scope_name=s.is_str(),
            channels=s.is_list(s.is_int()),
        )
    )
