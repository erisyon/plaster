from zest import zest
from plaster.tools.uniprot import uniprot


def zest_fasta_split():
    def it_returns_multiple_lines_and_headers():
        fasta_str = """
        >sp|P08069|IGF1R_HUMAN Insulin-like growth factor 1 receptor OS=Homo sapiens OX=9606 GN=IGF1R PE=1 SV=1
        MKSGSGGGSPTSLWGLLFLSAALSLWPTSGEICGPGIDIRNDYQQLKRLENCTVIEGYLH
        ILLISKAEDYRSYRFPKLTVITEYLLLFRVAGLESLGDLFPNLTVIRGWKLFYNYALVIF
        >sp|P1010|IGF1R_HUMAN something else OS=Homo sapiens OX=9606 GN=IGF1R PE=1 SV=1
        ZZZZZZZ
        AAAAAAA
        """
        groups = uniprot.fasta_split(fasta_str)
        assert groups == [
            (
                "sp|P08069|IGF1R_HUMAN Insulin-like growth factor 1 receptor OS=Homo sapiens OX=9606 GN=IGF1R PE=1 SV=1",
                "MKSGSGGGSPTSLWGLLFLSAALSLWPTSGEICGPGIDIRNDYQQLKRLENCTVIEGYLHILLISKAEDYRSYRFPKLTVITEYLLLFRVAGLESLGDLFPNLTVIRGWKLFYNYALVIF",
            ),
            (
                "sp|P1010|IGF1R_HUMAN something else OS=Homo sapiens OX=9606 GN=IGF1R PE=1 SV=1",
                "ZZZZZZZAAAAAAA",
            ),
        ]

    def it_raises_on_bad_fasta():
        fasta_str = """
        MKSGSGGGSPTSLWGLLFLSAALSLWPTSGEICGPGIDIRNDYQQLKRLENCTVIEGYLH
        """
        with zest.raises(ValueError) as e:
            uniprot.fasta_split(fasta_str)
        assert "data before the header" in str(e.exception)

    def it_returns_empty():
        groups = uniprot.fasta_split("")
        assert groups == []

    def it_returns_on_none():
        groups = uniprot.fasta_split(None)
        assert groups == []

    zest()
