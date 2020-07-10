from plaster.tools.utils import utils
from plaster.tools.utils.simple_http import http_method


def fasta_split(fasta_str):
    """
    Splits a fasta file like:
        >sp|P10636|TAU_HUMAN Microtubule-associated protein tau OS=Homo sapiens OX=9606 GN=MAPT PE=1 SV=5
        MAEPRQEFEVMEDHAGTYG
        SPRHLSNVSST
        >ANOTHER_TAU_HUMAN Microtubule-associated protein tau OS=Homo sapiens OX=9606 GN=MAPT PE=1 SV=5
        ABC DEF
        GHI

    Returns:
        List(Tuple(header, sequence))
    """
    if fasta_str is None:
        fasta_str = ""

    lines = fasta_str.split("\n")

    groups = []
    group = []
    last_header = None

    def close(last_header):
        nonlocal group, groups
        group = [g.strip() for g in group if g.strip() != ""]
        if last_header is None and len(group) > 0:
            raise ValueError("fasta had data before the header")
        if last_header is not None:
            groups += [(last_header[1:], "".join(group))]
        group = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            close(last_header)
            last_header = line
        else:
            group += [line]
    close(last_header)

    return groups


# Uniprot has a problem with their DNS and the easiest way around it for
# now is to hardcodfe their IP address.
# uniprot = "www.uniprot.org"
uniprot = "128.175.245.202"


@utils.cache()
def _get(url_path):
    return http_method(f"https://{uniprot}{url_path}", n_retries=5, allow_unverified=True)


def get_ac_fasta(ac):
    """
    Returns a List(tuple(header, seq)) of the resulting sequences, usually the list len == 1
    """
    return _get("/uniprot/{ac}.fasta")


def get_proteome(proteome_id):
    return _get("/uniprot/?include=false&format=fasta&force=true&query=proteome:{proteome_id}")
