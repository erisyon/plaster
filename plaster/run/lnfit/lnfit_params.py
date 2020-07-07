from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class LNFitParams(Params):
    defaults = Munch(photometry_only=False)

    schema = s(
        s.is_kws_r(
            dye_on_threshold=s.is_int(),
            photometry_only=s.is_bool(),
            lognormal_fitter_v2_params=s.is_str(),
        )
    )
