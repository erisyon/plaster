from zest import zest
import pandas as pd
import numpy as np
from plaster.run.error_model import ErrorModel
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v1.sim_v1_params import SimV1Params
from plaster.run.sim_v1.sim_v1_result import SimV1Result, DyeType
from plaster.tools.utils.utils import npf, np_array_same


def zest_sim_v1_result():
    def _make_sim_params(aa_list, n_edmans, n_train_samples=4, n_test_samples=4):
        params = SimV1Params.construct_from_aa_list(
            aa_list,
            error_model=ErrorModel.no_errors(n_channels=len(aa_list)),
            n_samples_train=n_train_samples,
            n_samples_test=n_test_samples,
            n_pres=1,
            n_mocks=0,
            n_edmans=n_edmans,
        )
        return params

    def mat_tests():
        n_peps = 2
        n_cycles = 3
        n_channels = 2
        n_train_samples = 4
        n_test_samples = 3

        result = None

        def _make_train_radmat():
            return npf([
                [[3.0, 2.0, 1.0], [1.0, 0.0, 1.0]],
                [[4.0, 2.0, 1.0], [1.0, 1.0, 0.0]],
                [[5.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
                [[6.0, 3.0, 0.9], [1.0, 0.0, 1.0]],
                [[0.3, 0.2, 0.1], [0.3, 0.2, 0.1]],
                [[0.4, 0.2, 0.1], [0.3, 0.2, 0.1]],
                [[0.5, 0.2, 0.1], [0.3, 0.2, 0.1]],
                [[0.6, 0.3, 2.9], [0.3, 0.2, 0.1]],
            ])

        def _make_test_radmat():
            return npf([
                [[4.0, 2.0, 1.0], [1.0, 1.0, 0.0]],
                [[5.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
                [[6.0, 3.0, 0.9], [1.0, 0.0, 1.0]],
                [[0.4, 0.2, 0.1], [0.3, 0.2, 0.1]],
                [[0.5, 0.2, 0.1], [0.3, 0.2, 0.1]],
                [[0.6, 0.3, 2.9], [0.3, 0.2, 0.1]],
            ])

        def _make_sim_result():
            sim_params = _make_sim_params(
                ["A", "B"],
                n_edmans=n_cycles - 1,
                n_train_samples=n_train_samples,
                n_test_samples=n_test_samples,
            )
            assert sim_params.n_channels == n_channels

            train_shape = (n_peps * n_train_samples, n_channels, n_cycles)
            test_shape = (n_peps * n_test_samples, n_channels, n_cycles)

            train_dyemat = np.zeros(shape=train_shape, dtype=np.uint8)
            train_radmat = _make_train_radmat()
            assert train_radmat.shape == train_shape

            test_dyemat = np.zeros(shape=test_shape, dtype=np.uint8)
            test_radmat = _make_test_radmat()
            assert test_radmat.shape == test_shape

            sim_result = SimV1Result(
                params=sim_params,
                train_dyemat=train_dyemat,
                train_radmat=train_radmat,
                train_true_pep_iz=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                test_dyemat=test_dyemat,
                test_radmat=test_radmat,
                test_true_pep_iz =np.array([0, 0, 0, 1, 1, 1]),
            )
            return sim_result

        def _before():
            nonlocal result
            result = _make_sim_result()

        def it_gets_flat_train_radmat():
            flat_train_radmat = result.flat_train_radmat()
            assert flat_train_radmat.shape == (
                n_peps * n_train_samples,
                n_channels * n_cycles,
            )
            expected = npf(
                [
                    [3.0, 2.0, 1.0, 1.0, 0.0, 1.0],
                    [4.0, 2.0, 1.0, 1.0, 1.0, 0.0],
                    [5.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                    [6.0, 3.0, 0.9, 1.0, 0.0, 1.0],
                    [0.3, 0.2, 0.1, 0.3, 0.2, 0.1],
                    [0.4, 0.2, 0.1, 0.3, 0.2, 0.1],
                    [0.5, 0.2, 0.1, 0.3, 0.2, 0.1],
                    [0.6, 0.3, 2.9, 0.3, 0.2, 0.1],
                ]
            )
            assert np_array_same(flat_train_radmat, expected)

        def it_gets_flat_test_radmat():
            flat_test_radmat = result.flat_test_radmat()
            assert flat_test_radmat.shape == (
                n_peps * n_test_samples,
                n_channels * n_cycles,
            )
            expected = npf(
                [
                    [4.0, 2.0, 1.0, 1.0, 1.0, 0.0],
                    [5.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                    [6.0, 3.0, 0.9, 1.0, 0.0, 1.0],
                    [0.4, 0.2, 0.1, 0.3, 0.2, 0.1],
                    [0.5, 0.2, 0.1, 0.3, 0.2, 0.1],
                    [0.6, 0.3, 2.9, 0.3, 0.2, 0.1],
                ]
            )
            assert np_array_same(flat_test_radmat, expected)

        def it_gets_train_true_pep_iz():
            pep_iz = result.train_true_pep_iz
            assert len(pep_iz) == n_peps * n_train_samples
            assert pep_iz.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

        def it_gets_test_true_pep_iz():
            pep_iz = result.test_true_pep_iz
            assert len(pep_iz) == n_peps * n_test_samples
            assert pep_iz.tolist() == [0, 0, 0, 1, 1, 1]

        zest()

    def flus_tests():
        result = None
        prep_result = None

        def _before():
            nonlocal result, prep_result
            sim_params = _make_sim_params(["AB", "CD"], n_edmans=3)
            prep_result = PrepResult.stub_prep_result(
                pros=[".", "ABCDAADD", "AXCABC"],
                pro_is_decoys=[False, False, False],
                peps=[".", "ABCD", "AADD", "AXCABC"],
                pep_pro_iz=[0, 1, 1, 2],
            )
            prep_result.params = PrepParams(
                proteins=[
                    dict(name="id_0", sequence="."),
                    dict(name="id_1", sequence="ABCDAADD"),
                    dict(name="id_2", sequence="AXCABC"),
                ]
            )
            result = SimV1Result(params=sim_params)
            result._generate_flu_info(prep_result)

        def it_gets_flus():
            flus = result.flus()
            assert len(flus) == 4
            assert flus.pep_i.tolist() == [0, 3, 1, 2]  # order is flustr
            assert flus.flustr.tolist() == [
                "... ;0,0",
                "0.1 ;2,1",
                "001 ;0,1",
                "001 ;0,1",
            ]
            assert flus.flu_count.tolist() == [1, 1, 2, 2]

        def it_gets_peps__flus():
            peps__flus = result.peps__flus(prep_result)
            assert len(peps__flus) == 4
            assert peps__flus.pep_i.tolist() == [0, 1, 2, 3]  # order is pep_i
            assert peps__flus.pro_i.tolist() == [0, 1, 1, 2]
            assert "pep_start" in peps__flus.columns
            assert "pep_stop" in peps__flus.columns
            assert peps__flus.flustr.tolist() == [
                "... ;0,0",
                "001 ;0,1",
                "001 ;0,1",
                "0.1 ;2,1",
            ]

        def it_gets_peps__flus__unique_flus():
            unique_flus = result.peps__flus__unique_flus(prep_result)
            assert len(unique_flus) == 2
            assert unique_flus.pep_i.tolist() == [0, 3]
            assert unique_flus.flustr.tolist() == ["... ;0,0", "0.1 ;2,1"]

        def it_gets_pros__peps__pepstrs__flus():
            pros_with_flus = result.pros__peps__pepstrs__flus(prep_result)
            assert pros_with_flus.pep_i.tolist() == [0, 1, 2, 3]
            assert pros_with_flus.pro_id.tolist() == ["id_0", "id_1", "id_1", "id_2"]
            assert pros_with_flus.seqstr.tolist() == [".", "ABCD", "AADD", "AXCABC"]
            assert pros_with_flus.flustr.tolist() == [
                "... ;0,0",
                "001 ;0,1",
                "001 ;0,1",
                "0.1 ;2,1",
            ]

        zest()

    zest()
