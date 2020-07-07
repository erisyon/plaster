from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class PrepParams(Params):
    defaults = Munch(
        protease=None,
        decoy_mode=None,
        include_misses=0,
        n_peps_limit=None,
        drop_duplicates=False,
        n_ptms_limit=None,
    )

    schema = s(
        s.is_kws_r(
            protease=s.is_list(noneable=True, elems=s.is_str()),
            decoy_mode=s.is_str(noneable=True),
            include_misses=s.is_int(),
            n_peps_limit=s.is_int(noneable=True),
            drop_duplicates=s.is_bool(),
            n_ptms_limit=s.is_int(noneable=True),
            proteins=s.is_list(
                s.is_kws(
                    name=s.is_str(required=True),
                    sequence=s.is_str(required=True),
                    ptm_locs=s.is_str(noneable=True),
                    report=s.is_int(noneable=True),
                    abundance=s.is_number(noneable=True),
                )
            ),
        )
    )
