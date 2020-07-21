import numpy as np
import pandas as pd
from munch import Munch
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.nn_v2.nn_v2_worker import nn_v2_worker
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel
from plaster.tools.log.log import debug, prof
from zest import zest


def zest_nn_v2_worker():
    import pudb; pudb.set_trace()
    prep_params = PrepParams(proteins=[Munch(name="pep1", sequence="ABCDE")])

    pros = pd.DataFrame(
        dict(
            pro_id=["nul", "pep1"],
            pro_is_decoy=[False, False],
            pro_i=[0, 1],
            pro_ptm_locs=[None, None],
            pro_report=[None, None],
        )
    )

    pro_seqs = pd.DataFrame(
        dict(pro_i=[0, 1, 1, 1, 1, 1], aa=[".", "A", "B", "C", "A", "A"],)
    )

    peps = pd.DataFrame(
        dict(pep_i=[0, 1, 2], pep_start=[0, 0, 3], pep_stop=[1, 3, 5], pro_i=[0, 1, 1],)
    )

    pep_seqs = pd.DataFrame(
        dict(
            pep_i=[0, 1, 1, 1, 2, 2],
            aa=[".", "A", "B", "C", "A", "A"],
            pep_offset_in_pro=[0, 0, 1, 2, 3, 4],
        )
    )

    prep_result = PrepResult(
        params=prep_params,
        _pros=pros,
        _pro_seqs=pro_seqs,
        _peps=peps,
        _pep_seqs=pep_seqs,
    )

    error_model = ErrorModel.no_errors(n_channels=2)

    sim_v2_params = SimV2Params.construct_from_aa_list(
        ["A", "B"], error_model=error_model, n_edmans=4
    )

    sim_v2_result = sim_v2_worker.sim(sim_v2_params, prep_result)

    nn_v2_params = NNV2Params()

    nn_v2_result = nn_v2_worker(nn_v2_params, sim_v2_result)

    def it_predicts_test():
        debug(nn_v2_result.test_pred_dt_iz)

    zest()
