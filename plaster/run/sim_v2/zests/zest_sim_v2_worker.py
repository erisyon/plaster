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


def zest_gen_flus():
    error_model = ErrorModel.no_errors(n_channels=2, p_non_fluorescent=0.5)

    sim_v2_params = SimV2Params.construct_from_aa_list(
        ["A", "B"], error_model=error_model, n_edmans=4
    )

    pep_seqs = pd.DataFrame(
        dict(
            pep_i=[0, 1, 1,  1, 2, 2],
            aa=[".", "A", "B",  "C", "A", "A"],
            pep_offset_in_pro=[0, 0, 1,  2, 3, 4],
        )
    )

    flus, pi_brights = sim_v2_worker._gen_flus(sim_v2_params, pep_seqs)

    def it_returns_flus():
        assert utils.np_array_same(flus[0], np.array([7], dtype=np.uint8))
        assert utils.np_array_same(flus[1], np.array([0, 1, 7], dtype=np.uint8))
        assert utils.np_array_same(flus[2], np.array([0, 0], dtype=np.uint8))

    def it_returns_p_bright():
        half_uint64_max = 9223372036854775807
        assert utils.np_array_same(pi_brights[0], np.array([0], dtype=np.uint64))
        assert utils.np_array_same(pi_brights[1], np.array([half_uint64_max, half_uint64_max, 0], dtype=np.uint64))
        assert utils.np_array_same(pi_brights[2], np.array([half_uint64_max, half_uint64_max], dtype=np.uint64))

    zest()


def zest_sim_v2_worker():
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

    pro_seqs = pd.DataFrame(dict(pro_i=[0, 1, 1, 1, 1, 1], aa=[".", "A", "B", "C", "A", "A"],))

    peps = pd.DataFrame(
        dict(pep_i=[0, 1, 2], pep_start=[0, 0, 3], pep_stop=[1, 3, 5], pro_i=[0, 1, 1],)
    )

    pep_seqs = pd.DataFrame(
        dict(
            pep_i=[0, 1, 1,  1, 2, 2],
            aa=[".", "A", "B",  "C", "A", "A"],
            pep_offset_in_pro=[0, 0, 1,  2, 3, 4],
        )
    )

    prep_result = PrepResult(
        params=prep_params,
        _pros=pros,
        _pro_seqs=pro_seqs,
        _peps=peps,
        _pep_seqs=pep_seqs,
    )

    def _sim(err_kwargs=None):
        error_model = ErrorModel.no_errors(n_channels=2, **err_kwargs)

        sim_v2_params = SimV2Params.construct_from_aa_list(
            ["A", "B"], error_model=error_model, n_edmans=4
        )

        return sim_v2_worker.sim(sim_v2_params, prep_result)

    def it_returns_train_dyemat():
        # Because it has no errors, there's only a perfect dyemats
        sim_v2_result = _sim()
        assert sim_v2_result.train_dyemat.shape == (3, 5 * 2)  # 5 cycles, 2 channels
        assert utils.np_array_same(
            sim_v2_result.train_dyemat[1:, :],
            np.array([
                [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.uint8)
        )

    def it_returns_train_dyemat_with_a_zero_row():
        sim_v2_result = _sim()
        assert np.all(sim_v2_result.train_dyemat[0, :] == 0)

    def it_returns_train_dyepeps():
        sim_v2_result = _sim()
        assert utils.np_array_same(
            sim_v2_result.train_dyepeps,
            np.array([
                [0, 0, 5000],
                [1, 1, 5000],
                [2, 2, 5000]
            ], dtype=np.uint64)
        )

    def it_handles_non_fluorescent():
        sim_v2_result = _sim(dict(p_non_fluorescent=0.5))
        # Check that every dyepep other than the nul-row
        # should have counts (col=2) a lot less than 5000.
        assert np.all(sim_v2_result.train_dyepeps[1:, 2] < 2000)

    def it_returns_no_all_dark_samples():
        sim_v2_result = _sim(dict(p_non_fluorescent=0.99))
        assert not np.any(sim_v2_result.train_dyepeps[1:, 0] == 0)

    def it_returns_train_recalls():
        sim_v2_result = _sim()
        raise NotImplementedError

    # def it_returns_train_flus():
    #     raise NotImplementedError
    #
    # def it_returns_train_flu_remainders():
    #     raise NotImplementedError
    #
    # def it_returns_test_radmat():
    #     raise NotImplementedError
    #
    # def it_returns_test_dyemat():
    #     raise NotImplementedError
    #
    # def it_returns_test_radmat_true_pep_iz():
    #     raise NotImplementedError


# it_gives_up_on_hard_peptides_and_returns_none
#
# def it_maintains_decoys_for_train():
# def it_removes_decoys_for_test():
# def it_raises_if_train_and_test_identical():

    zest()
