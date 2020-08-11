from zest import zest
from io import StringIO
import pandas as pd
import numpy as np
from plaster.tools.log.log import debug, prof
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel
from plaster.run.sim_v2.fast import sim_v2_fast
from plaster.tools.schema import check
from plaster.tools.utils import utils


def zest_pyx_runs():
    sim_params = SimV2Params.construct_from_aa_list(
        ["L", "Q"], error_model=ErrorModel.from_defaults(n_channels=2)
    )

    csv = """pep_i,aa,pep_offset_in_pro
        0,.,0
        1,M,0
        1,L,1
        1,K,2
        1,P,3
        2,N,15
        2,C,16
        2,Q,17
        2,R,18
    """
    pep_seq_df = pd.read_csv(StringIO(csv))

    labelled_pep_df = pd.merge(
        left=pep_seq_df,
        right=sim_params.df,
        left_on="aa",
        right_on="amino_acid",
        how="left",
    )

    labelled_pep_df.sort_values(by=["pep_i", "pep_offset_in_pro"], inplace=True)
    pcbs = labelled_pep_df[["pep_i", "ch_i", "p_bright"]].values

    # flus = []
    # pep_pi_brights = []
    # for pep_i, group in labelled_pep_df.groupby("pep_i"):
    #     flu_float = group.ch_i.values
    #     flu = np.nan_to_num(flu_float, nan=sim_v2_fast.NO_LABEL).astype(
    #         sim_v2_fast.DyeType
    #     )
    #     flus += [flu]
    #     pep_pi_brights += [
    #         np.full((len(flu_float)), 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
    #     ]

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
    dyetracks, dyepeps, pep_recalls = sim_v2_fast.sim(
        pcbs,
        sim_params.n_samples_train,
        sim_params.n_channels,
        cycles,
        sim_params.error_model.dyes[0].p_bleach_per_cycle,
        sim_params.error_model.p_detach,
        sim_params.error_model.p_edman_failure,
        n_threads=1,
        rng_seed=1,
    )

    check.array_t(dyetracks, dtype=np.uint8, shape=(5, 4))
    check.array_t(dyepeps, dtype=np.uint64, shape=(4, 3))

    def it_reserves_row_zero():
        assert utils.np_array_same(dyetracks[0], [0, 0, 0, 0])

    def it_has_no_row_all_zero_other_than_first():
        assert np.all(np.sum(dyetracks[1:], axis=1) > 0)

    def it_has_all_peps_in_dyepeps_except_nul():
        assert np.sort(np.unique(dyepeps[:, 1])).tolist() == [1, 2]

    def it_has_no_counts_on_the_nul_record():
        assert not np.any(dyepeps[:, 0] == 0)

    def it_has_all_counts_reasonable():
        assert np.all(dyepeps[:, 2] > 100)

    zest()
