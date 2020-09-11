from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class ImsImportParams(Params):
    defaults = Munch(
        is_movie=False,
        start_field=0,
        n_fields_limit=None,
        start_cycle=0,
        n_cycles_limit=None,
        dst_ch_i_to_src_ch_i=None,
        is_z_stack_single_file=False,
        z_stack_n_slices_per_field=None,
    )

    # Note that in movie mode what is called "field" is really the "frame" since the
    # stage does not move between shots.
    # The single .nd2 file in movie mode then treats the "fields" as if they are "cycles"
    # of a single field.

    schema = s(
        s.is_kws_r(
            is_movie=s.is_bool(),
            start_field=s.is_int(),
            n_fields_limit=s.is_int(noneable=True),
            start_cycle=s.is_int(),
            n_cycles_limit=s.is_int(noneable=True),
            dst_ch_i_to_src_ch_i=s.is_list(elems=s.is_int(), noneable=True),
            is_z_stack_single_file=s.is_bool(),
            z_stack_n_slices_per_field=s.is_int(noneable=True),
        )
    )
