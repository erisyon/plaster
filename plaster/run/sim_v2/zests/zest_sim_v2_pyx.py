import numpy as np
from plaster.tools.log.log import debug, prof
from plaster.run.sim_v2.sim_v2_params import SimV2Params

# from plaster.run.sim_v2 import csim_v2_fast
from plaster.run.prep.prep_result import PrepResult
from plaster.tools.utils import utils


def _zest_pyx_runs():
    which = "10"

    # TODO CONVERT to make this
    prep_result = PrepResult.load_from_folder(
        f"./jobs_folder/yoda_v2_classify_{which}/gluc_ph8_de_k_r_a88f/plaster_output/prep"
    )
    sim_params = utils.yaml_load_munch(
        f"./jobs_folder/yoda_v2_classify_{which}/gluc_ph8_de_k_r_a88f/plaster_run.yaml"
    )
    sim_params = utils.block_search(sim_params, "sim.parameters")
    sim_params = SimV2Params(include_dfs=True, **sim_params)

    flus = []
    pep_seq_df = prep_result.pepseqs()
    labelled_pep_df = pep_seq_df.join(
        sim_params.df.set_index("amino_acid"), on="aa", how="left"
    )
    for pep_i, group in labelled_pep_df.groupby("pep_i"):
        flu_float = group.ch_i.values
        flu = np.nan_to_num(flu_float, nan=csim_v2_fast.NO_LABEL).astype(
            csim_v2_fast.DyeType
        )
        flus += [flu]

    cycles = np.zeros((sim_params.n_cycles,), dtype=csim_v2_fast.CycleKindType)
    i = 0
    for _ in range(sim_params.n_pres):
        cycles[i] = csim_v2_fast.CYCLE_TYPE_PRE
        i += 1
    for _ in range(sim_params.n_mocks):
        cycles[i] = csim_v2_fast.CYCLE_TYPE_MOCK
        i += 1
    for _ in range(sim_params.n_edmans):
        cycles[i] = csim_v2_fast.CYCLE_TYPE_EDMAN
        i += 1

    # TODO: bleach each channel
    # TODO: Include the p_bright calculations yet
    prof()
    dyetracks, dyepeps = csim_v2_fast.sim(
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
    debug(dyetracks.shape)
    # TODO: Put asserts, it...
