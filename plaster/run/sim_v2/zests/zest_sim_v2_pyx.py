from io import StringIO
import pandas as pd
import numpy as np
from plaster.tools.log.log import debug, prof
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel
from plaster.run.sim_v2.fast import sim_v2_fast


def zest_pyx_runs():
    sim_params = SimV2Params.construct_from_aa_list(
        ["L", "Q"], error_model=ErrorModel.from_defaults(n_channels=2)
    )

    csv = """pep_i,aa,pep_offset_in_pro
        0, ., 0
        1, M, 0
        1, L, 1
        1, K, 2
        1, P, 3
        2, N, 15
        2, C, 16
        2, Q, 17
        2, R, 18
    """
    pep_seq_df = pd.read_csv(StringIO(csv))

    import pudb; pudb.set_trace()
    flus = []
    labelled_pep_df = pep_seq_df.join(
        sim_params.df.set_index("amino_acid"), on="aa", how="left"
    )
    for pep_i, group in labelled_pep_df.groupby("pep_i"):
        flu_float = group.ch_i.values
        flu = np.nan_to_num(flu_float, nan=sim_v2_fast.NO_LABEL).astype(
            sim_v2_fast.DyeType
        )
        flus += [flu]

    cycles = np.zeros((sim_params.n_cycles,), dtype=sim_v2_fast.CycleKindType)
    i = 0
    for _ in range(sim_params.n_pres):
        cycles[i] = sim_v2_fast.CYCLE_TYPE_PRE
        i += 1
    for _ in range(sim_params.n_mocks):
        cycles[i] = sim_v2_fast.CYCLE_TYPE_MOCK
        i += 1
    for _ in range(sim_params.n_edmans):
        cycles[i] = sim_v2_fast.CYCLE_TYPE_EDMAN
        i += 1

    # TODO: bleach each channel
    # TODO: Include the p_bright calculations yet
    prof()
    dyetracks, dyepeps = sim_v2_fast.sim(
        flus,
        sim_params.n_samples_train,
        sim_params.n_channels,
        cycles,
        sim_params.error_model.dyes[0].p_bleach_per_cycle,
        sim_params.error_model.p_detach,
        sim_params.error_model.p_edman_failure,
        n_threads=1,
        rng_seed=None,
    )
    prof()
    import pudb; pudb.set_trace()
    debug(dyetracks.shape)
    # TODO: Put asserts, it...
