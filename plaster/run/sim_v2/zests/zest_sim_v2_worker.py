from munch import Munch
import pandas as pd
import numpy as np
from plaster.tools.log.log import debug, prof
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v2.sim_v2_params import SimV2Params, ErrorModel
from plaster.run.sim_v2 import sim_v2_worker
from plaster.tools.utils import utils
from zest import zest


def zest_sim_v2_worker():
    prep_params = PrepParams(proteins=[Munch(name="pep1", sequence="ABCDE")])

    pros = pd.DataFrame(
        dict(
            pro_id=["pep1"],
            pro_is_decoy=[False],
            pro_i=[1],
            pro_ptm_locs=[None],
            pro_report=[None],
        )
    )

    pro_seqs = pd.DataFrame(dict(pro_i=[1, 1, 1, 1, 1], aa=["A", "B", "C", "A", "A"],))

    peps = pd.DataFrame(
        dict(pep_i=[1, 2], pep_start=[0, 3], pep_stop=[3, 5], pro_i=[1, 1],)
    )

    pep_seqs = pd.DataFrame(
        dict(
            pep_i=[1, 1, 1, 2, 2],
            aa=["A", "B", "C", "A", "A"],
            pep_offset_in_pro=[0, 1, 2, 3, 4],
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

    def it_returns_train_dyemat():
        # Because it has no errors, there's only a perfect dyemats
        assert sim_v2_result.train_dyemat.shape == (3, 5 * 2)  # 5 cycles, 2 channels
        assert utils.np_array_same(
            sim_v2_result.train_dyemat[1:, :],
            np.array(
                [[1, 0, 0, 0, 0, 1, 1, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.uint8,
            ),
        )

    def it_returns_train_dyemat_with_a_zero_row():
        assert np.all(sim_v2_result.train_dyemat[0, :] == 0)

    def it_returns_train_recalls():
        raise NotImplementedError

    def it_returns_train_flus():
        raise NotImplementedError

    def it_returns_train_flu_remainders():
        raise NotImplementedError

    def it_returns_train_dyepeps():
        raise NotImplementedError

    def it_returns_test_radmat():
        raise NotImplementedError

    def it_returns_test_dyemat():
        raise NotImplementedError

    def it_returns_test_radmat_true_pep_iz():
        raise NotImplementedError

    zest()
