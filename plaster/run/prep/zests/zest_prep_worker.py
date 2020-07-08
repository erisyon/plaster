import numpy as np
import pandas as pd
import itertools
from zest import zest
from plaster.run.prep import prep_worker
from plaster.run.prep.prep_worker import (
    _proteolyze,
    _step_1_check_for_uniqueness,
    _step_2_create_pros_and_pro_seqs_dfs,
    _step_3_generate_decoys,
    _step_4_proteolysis,
    _do_ptm_permutations,
    _step_5_create_ptm_peptides,
    _error,
)
from plaster.tools.aaseq import aaseq
from plaster.tools.log.log import debug


def _pro_seq_df(letters):
    df = pd.DataFrame([(1, let) for let in list(letters)], columns=["pro_i", "aa"])
    return df


def seqstrs_from_proteolyze(df):
    df = (
        df.groupby("pro_pep_i")
        .apply(lambda x: x.aa.str.cat())
        .reset_index()
        .set_index("pro_pep_i")
        .sort_index()
        .rename(columns={0: "seqstr"})
        .reset_index()
    )
    return df.seqstr.values.tolist()


def zest_proteolyze():

    protease = "lysc"  # Cuts after K
    protease_list = ["lysc", "endopro"]  # Cuts after K, Cuts after AP

    def it_cleaves_with_protease():
        df, seqstrs = (None,) * 2

        def _before():
            nonlocal df, seqstrs
            df = _proteolyze(_pro_seq_df("ABKCD"), protease)
            seqstrs = seqstrs_from_proteolyze(df)

        def it_picks_up_first():
            assert seqstrs == ["ABK", "CD"]

        def it_assigns_offsets():
            assert df.pep_offset_in_pro.values.tolist() == list(range(5))

        def it_handles_cut_at_end():
            df = _proteolyze(_pro_seq_df("ABKCDK"), protease)
            seqstrs = seqstrs_from_proteolyze(df)
            assert seqstrs == ["ABK", "CDK"]

        def it_handles_no_cut():
            df = _proteolyze(_pro_seq_df("ABCD"), protease)
            seqstrs = seqstrs_from_proteolyze(df)
            assert seqstrs == ["ABCD"]

        zest()

    def it_cleaves_with_multiple_proteases():
        df, seqstrs = (None,) * 2

        def _before():
            nonlocal df, seqstrs
            df = _proteolyze(_pro_seq_df("TBKCDPEF"), protease_list)
            seqstrs = seqstrs_from_proteolyze(df)

        def it_picks_up_first():
            assert seqstrs == ["TBK", "CDP", "EF"]

        def it_assigns_offsets():
            assert df.pep_offset_in_pro.values.tolist() == list(range(8))

        def it_handles_cut_at_end():
            # end cut is 2nd protease
            df = _proteolyze(_pro_seq_df("TBKCDP"), protease_list)
            seqstrs = seqstrs_from_proteolyze(df)
            assert seqstrs == ["TBK", "CDP"]
            # end cut is 1st protease
            df = _proteolyze(_pro_seq_df("TBPCDK"), protease_list)
            seqstrs = seqstrs_from_proteolyze(df)
            assert seqstrs == ["TBP", "CDK"]

        def it_handles_no_cut():
            df = _proteolyze(_pro_seq_df("TBCD"), protease_list)
            seqstrs = seqstrs_from_proteolyze(df)
            assert seqstrs == ["TBCD"]

        zest()

    def it_cleaves_without_protease():
        df = _proteolyze(_pro_seq_df("ABKCD"), None)
        seqstrs = seqstrs_from_proteolyze(df)
        assert seqstrs == ["ABKCD"]

    zest()


def zest_step_1_check_for_uniqueness():
    def it_passes():
        pro_spec_df = pd.DataFrame(
            dict(name=["name1", "name2"], sequence=["ABC", "DEF"])
        )
        _step_1_check_for_uniqueness(pro_spec_df)

    def it_raises_on_duplicate_names():
        with zest.raises(ValueError):
            pro_spec_df = pd.DataFrame(
                dict(name=["name1", "name1"], sequence=["ABC", "DEF"])
            )
            _step_1_check_for_uniqueness(pro_spec_df)

    def it_raises_on_duplicate_sequences():
        with zest.mock(_error) as m_error:
            with zest.raises(ValueError):
                pro_spec_df = pd.DataFrame(
                    dict(name=["name1", "name2"], sequence=["ABC", "ABC"])
                )
                _step_1_check_for_uniqueness(pro_spec_df)
        assert m_error.called()

    zest()


def zest_step_2_create_pros_and_pro_seqs_dfs():
    pro_spec_df = pd.DataFrame(
        dict(
            name=["name1", "name2"],
            sequence=["ABC", "DEF"],
            ptm_locs=["", ""],
            report=[1, 1],
        )
    )
    pros_df, pro_seqs_df = _step_2_create_pros_and_pro_seqs_dfs(pro_spec_df)

    def it_adds_nul_row():
        assert pros_df.pro_i.tolist() == [0, 1, 2]
        assert pro_seqs_df.aa.tolist()[0] == "."

    def it_sets_decoy_to_false():
        assert pros_df.pro_is_decoy.tolist() == [False] * 3

    def it_returns_a_proper_pro_seq_df():
        assert pro_seqs_df.pro_i.tolist() == [0, 1, 1, 1, 2, 2, 2]
        assert pro_seqs_df.aa.tolist() == [".", "D", "E", "F", "A", "B", "C"]
        # The seqs are now order by report _and_ name and are reversed

    def it_sets_proteins_report_for_all():
        assert pros_df.pro_report.tolist()[0] == 0
        assert pros_df.pro_report.tolist()[1:] == [1, 1]

    def it_sets_proteins_report_for_specified():
        pro_spec_df = pd.DataFrame(
            dict(
                name=["name1", "name2"],
                sequence=["ABC", "DEF"],
                ptm_locs=["", ""],
                report=[1, 0],
            )
        )
        pros_df, pro_seqs_df = _step_2_create_pros_and_pro_seqs_dfs(pro_spec_df)
        assert pros_df.pro_report.tolist() == [0, 1, 0]
        assert pros_df.pro_id.tolist() == ["nul", "name1", "name2"]
        assert pro_seqs_df[pro_seqs_df.pro_i == 1].aa.tolist() == ["A", "B", "C"]

    def it_sorts_proteins_report_first():
        pro_spec_df = pd.DataFrame(
            dict(
                name=["name1", "name2"],
                sequence=["ABC", "DEF"],
                ptm_locs=["", ""],
                report=[0, 1],
            )
        )
        pros_df, pro_seqs_df = _step_2_create_pros_and_pro_seqs_dfs(pro_spec_df)
        assert pros_df.pro_report.tolist() == [0, 1, 0]
        assert pros_df.pro_id.tolist() == ["nul", "name2", "name1"]
        assert pro_seqs_df[pro_seqs_df.pro_i == 1].aa.tolist() == ["D", "E", "F"]

    zest()


def zest_step_3_generate_decoys():
    pros_df, pro_seqs_df = None, None

    def _before():
        nonlocal pros_df, pro_seqs_df
        pros_df = pd.DataFrame(
            [("nul", False, 0, ""), ("id1", False, 1, ""), ("id2", False, 2, ""),],
            columns=["pro_id", "pro_is_decoy", "pro_i", "pro_ptm_locs"],
        )

        pro_seqs_df = pd.DataFrame(
            [(0, "."), (1, "B"), (1, "C"), (2, "D"), (2, "E"),], columns=["pro_i", "aa"]
        )

    def positive_reverse():
        decoys_df, decoy_seqs_df = _step_3_generate_decoys(
            pros_df, pro_seqs_df, decoy_mode="reverse"
        )

        def it_sets_is_decoy_true():
            assert decoys_df.pro_is_decoy.tolist() == [True, True]

        def it_skips_nul():
            assert decoys_df.pro_i.tolist() == [3, 4]

        def it_names_the_decoy():
            assert decoys_df.pro_id.tolist() == ["rev-id1", "rev-id2"]

        def it_reverses():
            assert decoy_seqs_df.pro_i.tolist() == [3, 3, 4, 4]
            assert decoy_seqs_df.aa.tolist() == ["C", "B", "E", "D"]

        zest()

    def positive_shuffle():
        # add a new longer protein entry we can really test the shuffle with
        nonlocal pros_df, pro_seqs_df
        new_pro = {
            "pro_id": "id3",
            "pro_is_decoy": False,
            "pro_i": 3,
            "pro_ptm_locs": "",
        }
        pros_df = pros_df.append(new_pro, ignore_index=True)

        aas = sorted(list(aaseq.aa_code_df.aa))  # 27 chars long
        tups = list(itertools.zip_longest([], aas, fillvalue=3))
        new_pro_seqs = pd.DataFrame(tups, columns=["pro_i", "aa"])
        pro_seqs_df = pro_seqs_df.append(new_pro_seqs).reset_index(drop=True)

        decoys_df, decoy_seqs_df = _step_3_generate_decoys(
            pros_df, pro_seqs_df, decoy_mode="shuffle"
        )

        def it_sets_is_decoy_true():
            assert decoys_df.pro_is_decoy.tolist() == [True, True, True]

        def it_skips_nul():
            assert decoys_df.pro_i.tolist() == [4, 5, 6]

        def it_names_the_decoy():
            assert decoys_df.pro_id.tolist() == ["shu-id1", "shu-id2", "shu-id3"]

        def it_shuffles():
            assert decoy_seqs_df.pro_i.tolist() == [4, 4, 5, 5] + [6] * len(aas)
            assert decoy_seqs_df.aa.tolist()[4:] != aas
            assert decoy_seqs_df.aa.tolist()[4:] != aas[::-1]
            assert sorted(decoy_seqs_df.aa.tolist()[0:4]) == ["B", "C", "D", "E"]
            assert sorted(decoy_seqs_df.aa.tolist()[4:]) == aas

        zest()

    def it_handles_no_decoy_mode():
        decoys_df, decoy_seqs_df = _step_3_generate_decoys(
            pros_df, pro_seqs_df, decoy_mode=None
        )
        assert isinstance(decoys_df, pd.DataFrame) and len(decoys_df) == 0
        assert isinstance(decoy_seqs_df, pd.DataFrame) and len(decoy_seqs_df) == 0

    def it_reverses_ptm_locs():
        pros_df = pd.DataFrame(
            [("nul", False, 0, ""), ("id1", False, 1, ""), ("id2", False, 2, "1;3"),],
            columns=["pro_id", "pro_is_decoy", "pro_i", "pro_ptm_locs"],
        )

        pro_seqs_df = pd.DataFrame(
            [(0, "."), (1, "B"), (1, "C"), (2, "D"), (2, "E"), (2, "F"), (2, "G"),],
            columns=["pro_i", "aa"],
        )

        decoys_df, decoy_seqs_df = _step_3_generate_decoys(
            pros_df, pro_seqs_df, decoy_mode="reverse"
        )

        # remember that the orgiinal df has the "null" entry at iloc 0, and no
        # decoy is created for that.
        assert decoys_df.iloc[0].pro_ptm_locs == ""
        assert decoys_df.iloc[1].pro_ptm_locs == "2;4"

    zest()


def zest_step_4_proteolysis_one_protease():
    pro_seqs_df = pd.DataFrame(
        [(0, "."), (1, "B"), (1, "K"), (1, "C"), (2, "D"), (2, "E"),],
        columns=["pro_i", "aa"],
    )

    peps_df, pep_seqs_df = _step_4_proteolysis(pro_seqs_df, "lysc")  # Cut after K

    def it_renumbers_consecutively():
        assert peps_df.pro_i.tolist() == [0, 1, 1, 2]
        assert peps_df.pep_i.tolist() == [0, 1, 2, 3]

    def it_sets_seqs():
        assert pep_seqs_df.pep_i.tolist() == [0, 1, 1, 2, 3, 3]
        assert pep_seqs_df.pep_offset_in_pro.tolist() == [0, 0, 1, 2, 0, 1]
        assert pep_seqs_df.aa.tolist() == [".", "B", "K", "C", "D", "E"]

    zest()


def zest_step_4_proteolysis_multiple_proteases():
    pro_seqs_df = pd.DataFrame(
        [
            (0, "."),
            (1, "B"),
            (1, "K"),
            (1, "C"),
            (2, "D"),
            (2, "E"),
            (2, "K"),
            (2, "F"),
            (2, "G"),
            (2, "P"),
            (2, "H"),
        ],
        columns=["pro_i", "aa"],
    )

    peps_df, pep_seqs_df = _step_4_proteolysis(
        pro_seqs_df, ["lysc", "endopro"]
    )  # Cut after K, Cut after AP

    def it_renumbers_consecutively():
        assert peps_df.pro_i.tolist() == [0, 1, 1, 2, 2, 2]
        assert peps_df.pep_i.tolist() == [0, 1, 2, 3, 4, 5]

    def it_sets_seqs():
        assert pep_seqs_df.pep_i.tolist() == [0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5]
        assert pep_seqs_df.pep_offset_in_pro.tolist() == [
            0,
            0,
            1,
            2,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ]
        assert pep_seqs_df.aa.tolist() == [
            ".",
            "B",
            "K",
            "C",
            "D",
            "E",
            "K",
            "F",
            "G",
            "P",
            "H",
        ]

    zest()


def zest_do_ptm_permutations():
    df = pd.DataFrame(
        dict(
            pep_i=[1, 1, 1, 1, 1, 1,],
            aa=list("ABCDEF"),
            pep_offset_in_pro=list(range(6)),
            pro_ptm_locs=["1;4"] * 6,
        )
    )
    n_ptms_limit = 5

    zest.stack_mock(prep_worker._info)

    def it_adds_permutation_labels():
        new_pep_seqs = _do_ptm_permutations(df, n_ptms_limit)
        assert len(new_pep_seqs) == 3
        # remember the pro_ptm_locs string is 1-based
        assert new_pep_seqs[0].aa.tolist() == ["A[p]", "B", "C", "D", "E", "F"]
        assert new_pep_seqs[1].aa.tolist() == ["A", "B", "C", "D[p]", "E", "F"]
        assert new_pep_seqs[2].aa.tolist() == ["A[p]", "B", "C", "D[p]", "E", "F"]
        assert np.all(np.isnan(new_pep_seqs[0].pep_i.values))

    def it_handles_empty_ptm_locations():
        df_empty = df.copy()
        df_empty.pro_ptm_locs = ""
        new_pep_seqs = _do_ptm_permutations(df_empty, n_ptms_limit)
        assert len(new_pep_seqs) == 0

    def it_handles_too_many_permutations():
        # we allow 5 per peptide at present
        df_too_many = df.copy()
        df_too_many.pro_ptm_locs = "1;2;3;4;5;6"
        new_pep_seqs = _do_ptm_permutations(df_too_many, n_ptms_limit)
        assert len(new_pep_seqs) == 0

    zest()


def zest_step_5_create_ptm_peptides():
    peps_df = pd.DataFrame(
        [(0, 0, 1, 0), (1, 0, 5, 1),],
        columns=["pep_i", "pep_start", "pep_stop", "pro_i"],
    )
    pep_seqs_df = pd.DataFrame(
        dict(pep_i=[1] * 6, aa=list("ABCDEF"), pep_offset_in_pro=list(range(6)),)
    )
    pros_df = pd.DataFrame(
        [("nul", False, 0, ""), ("id1", False, 1, "1;3"),],
        columns=["pro_id", "pro_is_decoy", "pro_i", "pro_ptm_locs"],
    )

    def it_adds_correct_pep_iz():
        ptm_peps_df, ptm_pep_seqs_df = _step_5_create_ptm_peptides(
            peps_df, pep_seqs_df, pros_df, n_ptms_limit=5
        )
        assert len(ptm_peps_df) == 3
        assert ptm_peps_df.iloc[0].pep_i == 2
        assert ptm_peps_df.iloc[1].pep_i == 3
        assert ptm_peps_df.iloc[2].pep_i == 4

        assert len(ptm_pep_seqs_df) == 18
        assert list(ptm_pep_seqs_df.pep_i.unique()) == [2, 3, 4]

    zest()
