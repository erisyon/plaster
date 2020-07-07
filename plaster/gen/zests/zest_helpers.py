import numpy as np
from zest import zest
from plaster.gen import helpers
from plaster.tools.log.log import debug


def zest_protein_fasta():
    def it_parses():
        multi_fasta = """
        >sp|P10636|TAU_HUMAN Microtubule-associated protein tau OS=Homo sapiens OX=9606 GN=MAPT PE=1 SV=5
        MAEPRQEFEVMEDHAGTYG
        SPRHLSNVSST
        >ANOTHER_TAU_HUMAN Microtubule-associated protein tau OS=Homo sapiens OX=9606 GN=MAPT PE=1 SV=5
        ABC DEF
        GHI
        """

        proteins = helpers.protein_fasta(multi_fasta, None)
        assert proteins == [
            dict(seqstr="MAEPRQEFEVMEDHAGTYGSPRHLSNVSST", id="P10636"),
            dict(seqstr="ABC DEFGHI", id="ANOTHER_TAU_HUMAN"),
        ]

    zest()


def zest_protein_csv():
    zest.stack_mock(helpers._protein_csv_info)

    def it_accepts_name_and_sequence():
        csv_string = """
        Name, Seq
        P1, ABC
        P2, DEF
        """
        df = helpers.protein_csv_df(csv_string)
        assert df.loc[0, "seqstr"] == "ABC" and df.loc[0, "id"] == "P1"
        assert df.loc[1, "seqstr"] == "DEF" and df.loc[1, "id"] == "P2"

        def it_defaults_ptm_locs_to_empty_str():
            assert df.loc[0, "ptm_locs"] == ""

        def it_defaults_abundance_to_nan():
            assert np.isnan(df.loc[0, "abundance"])

        zest()

    def it_raises_if_no_seq_nor_uniprot_ac():
        csv_string = """
        Name, Abundance
        A, 10
        """
        with zest.raises(ValueError) as e:
            helpers.protein_csv_df(csv_string)
        assert "missing either a Seq or a UniprotAC" in str(e.exception)

    def it_raises_if_both_seq_and_uniprot_ac():
        csv_string = """
        Name, Seq, UniprotAC
        P1, A, P100
        """
        with zest.raises(ValueError) as e:
            helpers.protein_csv_df(csv_string)
        assert "both a Seq and a UniprotAC" in str(e.exception)

    def it_raises_if_no_name_and_no_uniprot_ac():
        csv_string = """
        Seq, Abundance
        ABC, 10
        """
        with zest.raises(ValueError) as e:
            helpers.protein_csv_df(csv_string)
        assert "missing a Name column" in str(e.exception)

    def it_reverse_sorts_by_abundance():
        csv_string = """
        Name, Seq, Abundance
        P1, ABC, 10
        P2, DEF, 100
        """
        df = helpers.protein_csv_df(csv_string)
        assert (
            df.loc[0, "seqstr"] == "DEF"
            and df.loc[0, "id"] == "P2"
            and df.loc[0, "abundance"] == 100.0
        )
        assert (
            df.loc[1, "seqstr"] == "ABC"
            and df.loc[1, "id"] == "P1"
            and df.loc[1, "abundance"] == 10.0
        )

    def it_sorts_by_name_if_no_abundance():
        csv_string = """
        Name, Seq
        P2, DEF
        P1, ABC
        """
        df = helpers.protein_csv_df(csv_string)
        assert df.loc[0, "seqstr"] == "ABC" and df.loc[0, "id"] == "P1"
        assert df.loc[1, "seqstr"] == "DEF" and df.loc[1, "id"] == "P2"

    def it_imports_ptm():
        csv_string = """
        Name, Seq, PTM
        P1, ABC, 3
        P2, DEF, 1;2
        P3, GHI, 
        """
        df = helpers.protein_csv_df(csv_string)
        assert df.loc[0, "ptm_locs"] == "3"
        assert df.loc[1, "ptm_locs"] == "1;2"
        assert df.loc[2, "ptm_locs"] == ""

    def it_lookups_uniprot():
        csv_string = """
        UniprotAC, Abundance
        P1, 10
        """
        with zest.mock(helpers._uniprot_lookup) as m:
            m.returns([{"id:": "foo", "seqstr": "ABC"}])
            df = helpers.protein_csv_df(csv_string)
        assert df.loc[0, "seqstr"] == "ABC"

        def it_uses_uniprot_ac_as_name():
            assert df.loc[0, "id"] == "P1"

        def it_imports_abundance():
            assert df.loc[0, "abundance"] == 10.0

        zest()

    def it_nans_missing_abundances():
        csv_string = """
        UniprotAC
        P1
        """
        with zest.mock(helpers._uniprot_lookup) as m:
            m.returns([{"id:": "foo", "seqstr": "ABC"}])
            df = helpers.protein_csv_df(csv_string)
        assert (
            df.loc[0, "id"] == "P1"
            and df.loc[0, "seqstr"] == "ABC"
            and np.isnan(df.loc[0, "abundance"])
        )

    def it_warns_on_no_seq_from_uniprot():
        csv_string = """
        UniprotAC
        P1
        """
        with zest.mock(helpers._protein_csv_warning) as m_warn:
            with zest.mock(helpers._uniprot_lookup) as m_lookup:
                m_lookup.returns([])
                helpers.protein_csv_df(csv_string)
        assert m_warn.called_once()

    def it_warns_on_more_than_one_seq_from_uniprot():
        csv_string = """
        UniprotAC
        P1
        """
        with zest.mock(helpers._protein_csv_warning) as m_warn:
            with zest.mock(helpers._uniprot_lookup) as m_lookup:
                m_lookup.returns(
                    [{"id": "foo", "seqstr": "123"}, {"id": "bar", "seqstr": "123456"}]
                )
                df = helpers.protein_csv_df(csv_string)
                assert len(df) == 1 and df.loc[0, "seqstr"] == "123456"
        assert m_warn.called_once()

    def it_raises_on_duplicate_names():
        csv_string = """
        Name, Seq
        P1, ABC
        P1, DEF
        """
        with zest.raises(ValueError) as e:
            helpers.protein_csv_df(csv_string)
        assert "duplicate names" in str(e.exception)

    def it_raises_on_duplicate_seqs():
        csv_string = """
        Name, Seq
        P1, ABC
        P2, ABC
        """
        with zest.raises(ValueError) as e:
            helpers.protein_csv_df(csv_string)
        assert "duplicate seqs" in str(e.exception)

    zest()
