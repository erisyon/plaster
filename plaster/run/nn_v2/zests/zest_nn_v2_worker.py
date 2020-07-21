import numpy as np
import pandas as pd
from munch import Munch
from plaster.run.sim_v2 import sim_v2_worker
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_result import PrepResult
from plaster.tools.log.log import debug, prof
from zest import zest


def zest_nn_v2_worker():
    nn_v2_params = NNV2Params()

    prep_params = PrepParams(proteins=[Munch(name="pep1", sequence="ABCDE")])

    pros = pd.DataFrame(
        dict(pro_i=[1], pro_id=["pep1"], pro_ptm_locs=[None], pro_report=[None],)
    )

    pro_seqs = pd.DataFrame(
        dict(
            aa=["A", "B", "C", "D", "E"],
            pro_i=[1, 1, 1, 1, 1],
            pro_name=["pep1", "pep1", "pep1", "pep1", "pep1"],
            pro_ptm_locs=[None, None, None, None, None],
            pro_report=[None, None, None, None, None],
        )
    )

    prep_result = PrepResult(
        params=prep_params,
        _pros=pros,
        _pro_seqs=pro_seqs,
        _peps=pros,
        _pep_seqs=pro_seqs,
    )

    # sim_v2_params = None  # TODO
    # sim_v2_result = sim_v2_worker.sim(sim_v2_params, prep_result)

    # TODO
    # sim_v2_result = SimV2Result(
    #     params=sim_v2_params,
    #     train_flus=train_flus,
    #     train_dyemat=train_dyemat,
    #     train_dyepeps_df=train_dyepeps_df,
    #     train_radmat=train_radmat,
    #     train_radmat_true_pep_iz=train_radmat_true_pep_iz,
    #     test_flus=test_flus,
    #     test_dyemat=test_dyemat,
    #     test_dyepeps_df=test_dyepeps_df,
    #     test_radmat=test_radmat,
    #     test_radmat_true_pep_iz=test_radmat_true_pep_iz,
    # )

    # nn_v2_fast.nn(nn_v2_params, prep_result, None)
    # nn_worker(nn_v2_params, prep_result, sim_v2_result)