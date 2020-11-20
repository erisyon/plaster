import itertools

import numpy as np
import pandas as pd
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_result import PrepResult
from plaster.tools.log import log
from zest import zest


def zest_PrepResult():
    def _make_prep_result(pros, is_decoys=[], abundances=[], ptm_locs=[], in_report=[]):
        """
        Builds a PrepResult from a list of proteins. Each protein can be a
        string or a list of strings. Lists of strings are returned as multiple
        peps for the same protein.
        Eg. _make_prep_result([("AAXC", "XKP"), "AGGH"]) will yield 2 pros and 3 peps.
        """
        n_pros = len(pros)
        names = [f"id_{i}" for i in range(n_pros)]
        seqstrs = ["".join(pro) for pro in pros]

        def _make_protein(name, seq, abundance, report):
            protein = dict(name=name, sequence=seq, report=report or 0)
            if abundance is not None:
                protein["abundance"] = abundance
            return protein

        proteins = [
            _make_protein(*params)
            for params in itertools.zip_longest(names, seqstrs, abundances, in_report)
        ]
        # PrepParams will log if abundance data has to be normalized. Silence those logs in the tests by mocking log
        with zest.mock(log.info) as m_log:
            params = PrepParams(proteins=proteins)

        _pros = pd.DataFrame(
            [
                (name, is_decoy or False, i, ptm_locs or "", in_report or 0)
                for i, (name, is_decoy, ptm_locs, in_report) in enumerate(
                    itertools.zip_longest(names, is_decoys, ptm_locs, in_report)
                )
            ],
            columns=PrepResult.pros_columns,
        )

        _pro_seqs = pd.DataFrame(
            [
                (pro_i, aa)
                for pro_i, seqstr in enumerate(seqstrs)
                for aa in list(seqstr)
            ],
            columns=PrepResult.pro_seqs_columns,
        )

        # normalize pros as lists of strings
        pros = [pro if isinstance(pro, list) else list(pro) for pro in pros]

        # extract peps from pros definitions
        peps_lens = [list(map(len, pro)) for pro in pros]
        peps = [
            (i, pep, start, stop - 1)
            for i, (pro, pep_lens) in enumerate(zip(pros, peps_lens))
            for pep, start, stop in zip(
                pro,
                itertools.accumulate([0] + pep_lens),
                itertools.accumulate(pep_lens),
            )
        ]

        _peps = pd.DataFrame(
            [(i, start, stop, pro_i) for i, (pro_i, _, start, stop) in enumerate(peps)],
            columns=PrepResult.peps_columns,
        )

        _pep_seqs = pd.DataFrame(
            [
                (pep_i, aa, start + offset)
                for pep_i, (_, pep, start, _) in enumerate(peps)
                for offset, aa in enumerate(list(pep))
            ],
            columns=PrepResult.pep_seqs_columns,
        )

        return PrepResult(
            params=params,
            _pros=_pros,
            _pro_seqs=_pro_seqs,
            _peps=_peps,
            _pep_seqs=_pep_seqs,
        )

    def pros():
        result = None

        def _before():
            nonlocal result

            result = _make_prep_result(
                [".", ["ABCDE", "FGHI"], "DDD"],
                is_decoys=[False, False, True],
                ptm_locs=["", "2;4", ""],
            )

        def it_gets_a_dataframe_with_reset_index():
            pros = result.pros()
            assert isinstance(pros, pd.DataFrame)
            assert pros.index.tolist() == list(range(len(pros)))

        def it_includes_abundances_if_in_params_proteins():
            result_with_abundances = _make_prep_result(
                [".", ["ABC", "DEF"], "XXD"], abundances=[0, 5, 10]
            )
            assert "abundance" in result_with_abundances.pros().columns
            # Note that abundances get normalized
            assert np.allclose(
                result_with_abundances.pros().abundance.tolist(),
                [0, 5, 10],
                equal_nan=True,
            )

        def it_gets_n_pros():
            assert result.n_pros == 3

        def it_gets_pros_abundance():
            result = _make_prep_result(
                [".", ["ABC", "DEF"], "XXD"], abundances=[0, 5, 10]
            )
            pros_abundance = result.pros_abundance()
            assert len(pros_abundance) == result.n_pros
            assert np.array_equal(pros_abundance, [0, 5, 10])

        def it_raises_on_nan_abundance():
            with zest.raises():
                result = _make_prep_result(
                    [".", ["ABC", "DEF"], "XXD"], abundances=[math.nan, 5, 10]
                )

        def it_gets_pros__ptm_locs():
            with_ptm_locs = result.pros__ptm_locs()
            assert len(with_ptm_locs) == 1
            assert with_ptm_locs.columns.tolist() == PrepResult.pros_columns
            assert with_ptm_locs.pro_ptm_locs.tolist() == ["2;4"]

        def it_gets_pros__from_decoys():
            from_decoys = result.pros__from_decoys()
            assert len(from_decoys) == 1
            assert from_decoys.columns.tolist() == PrepResult.pros_columns
            assert from_decoys.pro_id.tolist() == ["id_2"]

        def it_gets_pros__no_decoys():
            no_decoys = result.pros__no_decoys()
            assert len(no_decoys) == 2
            assert no_decoys.columns.tolist() == PrepResult.pros_columns
            assert no_decoys.pro_id.tolist() == ["id_0", "id_1"]

        def it_gets_proseqs():
            proseqs = result.proseqs()
            assert proseqs.query("pro_i == 1").aa.str.cat() == "ABCDEFGHI"
            assert proseqs.columns.tolist() == PrepResult.pro_seqs_columns

        def it_gets_prostrs():
            prostrs = result.prostrs()
            assert len(prostrs) == result.n_pros
            assert "seqstr" in prostrs.columns
            assert prostrs.seqstr.tolist() == [".", "ABCDEFGHI", "DDD"]

        zest()

    def pros_of_interest():
        result = None

        def _before():
            nonlocal result
            result = _make_prep_result([".", ["AB", "CD"], ["AA", "CB"]])

        def it_sets_pros_of_interest():
            result.set_pros_of_interest("id_0")
            assert result.pros().pro_report.astype(bool).tolist() == [
                True,
                False,
                False,
            ]
            result.set_pros_of_interest("id_1")
            assert result.pros().pro_report.astype(bool).tolist() == [
                False,
                True,
                False,
            ]
            result.set_pros_of_interest(["id_1", "id_2"])
            assert result.pros().pro_report.astype(bool).tolist() == [False, True, True]

        def it_gets_pros__in_report():
            result.set_pros_of_interest("id_1")
            in_report = result.pros__in_report()
            assert len(in_report) == 1
            assert in_report.set_index("pro_id").index.tolist() == ["id_1"]

        def it_gets_n_pros_of_interest():
            assert result.n_pros_of_interest == 0
            result.set_pros_of_interest(["id_0", "id_1"])
            assert result.n_pros_of_interest == 2

        def it_asserts_protein_ids():
            with zest.raises(AssertionError):
                result.set_pros_of_interest("P1")

        def it_sets_pro_ptm_locs():
            result.set_pro_ptm_locs("id_0", "1;3")
            assert result.pros().set_index("pro_id").at["id_0", "pro_ptm_locs"] == "1;3"
            assert "id_0" in result.pros__ptm_locs().pro_id.values

        def it_gets_pro_ptm_locs():
            assert result.get_pro_ptm_locs("id_0") == ""
            result.set_pro_ptm_locs("id_0", "1;3")
            assert result.get_pro_ptm_locs("id_0") == "1;3"

        zest()

    def peps():
        result = None

        def _before():
            nonlocal result

            result = _make_prep_result(
                [".", ["ABA", "CD"], ["ABA", "CB"]],
                abundances=[0, 5, 10],
                is_decoys=[True, False, True],
            )

        def it_gets_peps():
            peps = result.peps()
            assert isinstance(peps, pd.DataFrame)
            assert peps.columns.tolist() == PrepResult.peps_columns
            assert peps.pro_i.tolist() == [0, 1, 1, 2, 2]

        def it_gets_n_peps():
            assert result.n_peps == 5

        def it_gets_pepstrs():
            pepstrs = result.pepstrs()
            assert pepstrs.seqstr.values.tolist() == [".", "ABA", "CD", "ABA", "CB"]

        def it_gets_pepseqs():
            pepseqs = result.pepseqs()
            assert pepseqs.columns.tolist() == PrepResult.pep_seqs_columns
            assert pepseqs.aa.str.cat() == ".ABACDABACB"
            assert len(pepseqs) == 11

        def it_gets_peps_abundance():
            peps_abundance = result.peps_abundance()
            assert len(peps_abundance) == 5

            def it_fills_zero_for_missing_abundance():
                assert peps_abundance.tolist() == [0.0, 5.0, 5.0, 10.0, 10.0]

            zest()

        def it_gets_peps__no_decoys():
            no_decoys = result.peps__no_decoys()
            assert len(no_decoys) == 2
            assert no_decoys.pep_i.tolist() == [1, 2]
            assert no_decoys.pro_i.tolist() == [1, 1]
            assert no_decoys.columns.tolist() == PrepResult.peps_columns

            def it_handles_empty_return():
                result = _make_prep_result(["ABCDE", "DDE"], is_decoys=[True, True])
                assert len(result.peps__no_decoys()) == 0

            zest()

        def it_gets_peps__from_decoys():
            from_decoys = result.peps__from_decoys()
            assert len(from_decoys) == 3
            assert from_decoys.pep_i.tolist() == [0, 3, 4]
            assert from_decoys.pro_i.tolist() == [0, 2, 2]
            assert from_decoys.columns.tolist() == PrepResult.peps_columns

            def it_handles_empty_return():
                result = _make_prep_result(["ABC", "DEF"])
                assert len(result.peps__from_decoys()) == 0

            zest()

        def peps__in_report():
            result.set_pros_of_interest("id_1")
            in_report = result.peps__in_report()
            assert in_report.columns.tolist() == PrepResult.peps_columns
            assert len(in_report) == 2
            assert in_report.pro_i.drop_duplicates().tolist() == [1]

            def it_handles_empty_return():
                result.set_pros_of_interest([])
                assert len(result.peps__in_report()) == 0

            zest()

        def it_gets_peps__pepseqs():
            pepseqs = result.peps__pepseqs()
            assert len(pepseqs) == 11
            assert pepseqs.aa.str.cat() == ".ABACDABACB"

        def it_gets_pepseqs__no_decoys():
            pepseqs = result.pepseqs__no_decoys()
            assert len(pepseqs) == 5
            assert pepseqs.aa.str.cat() == "ABACD"

        def peps__ptms():
            result = _make_prep_result(
                [".", ["ABCD", "EGGX"], ["AABX", "HGK"]],
                is_decoys=[False, True, False],
                ptm_locs=["", "3;6", "1;3"],
            )
            peps__ptms = result.peps__ptms(ptms_to_rows=False)
            assert len(peps__ptms) == 1
            assert peps__ptms.at[0, "pro_id"] == "id_2"
            assert peps__ptms.at[0, "n_pep_ptms"] == 2
            assert peps__ptms.at[0, "pro_ptm_locs"] == "1;3"

            def it_filters_decoys():
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 3
                assert peps__ptms["pro_id"].tolist() == ["id_1", "id_1", "id_2"]

                peps__ptms = result.peps__ptms(
                    include_decoys=False,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 1
                assert peps__ptms["pro_id"].tolist() == ["id_2"]

            def it_filters_in_report_only():
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=True,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 0

                result.set_pros_of_interest("id_1")
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=True,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 2
                assert peps__ptms["pro_id"].tolist() == ["id_1", "id_1"]

            def it_filters_ptm_peps_only():
                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=False,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 4
                assert peps__ptms["pro_id"].tolist() == ["id_1", "id_1", "id_2", "id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["3", "6", "1;3", ""]

                peps__ptms = result.peps__ptms(
                    include_decoys=True,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 3
                assert peps__ptms["pro_id"].tolist() == ["id_1", "id_1", "id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["3", "6", "1;3"]

            def it_ptms_to_rows():
                peps__ptms = result.peps__ptms(
                    include_decoys=False,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=False,
                )
                assert len(peps__ptms) == 1
                assert peps__ptms["pro_id"].tolist() == ["id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["1;3"]

                peps__ptms = result.peps__ptms(
                    include_decoys=False,
                    in_report_only=False,
                    ptm_peps_only=True,
                    ptms_to_rows=True,
                )
                assert len(peps__ptms) == 2
                assert peps__ptms["pro_id"].tolist() == ["id_2", "id_2"]
                assert peps__ptms["pro_ptm_locs"].tolist() == ["1;3", "1;3"]
                assert peps__ptms["ptm"].tolist() == ["1", "3"]

            zest()

        zest()

    def pros_and_peps():
        result = _make_prep_result(
            [".", ["ABCD", "EF"], ["ABA", "CAB"]], is_decoys=[False, True, False],
        )

        def it_gets_pros__peps():
            pros__peps = result.pros__peps()
            assert len(pros__peps) == 5
            assert pros__peps["pro_id"].tolist() == [
                "id_0",
                "id_1",
                "id_1",
                "id_2",
                "id_2",
            ]
            assert pros__peps["pro_i"].tolist() == [0, 1, 1, 2, 2]
            assert pros__peps["pep_i"].tolist() == list(range(5))
            assert pros__peps["pro_is_decoy"].tolist() == [
                False,
                True,
                True,
                False,
                False,
            ]

        def it_gets_pros__peps__pepstrs():
            pros__peps__pepstrs = result.pros__peps__pepstrs()
            assert len(pros__peps__pepstrs) == 5
            assert pros__peps__pepstrs["pro_id"].tolist() == [
                "id_0",
                "id_1",
                "id_1",
                "id_2",
                "id_2",
            ]
            assert pros__peps__pepstrs["seqstr"].tolist() == [
                ".",
                "ABCD",
                "EF",
                "ABA",
                "CAB",
            ]

        zest()

    zest()
