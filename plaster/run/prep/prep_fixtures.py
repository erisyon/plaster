import random
from munch import Munch
import pandas as pd
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_result import PrepResult
from plaster.tools.aaseq.aaseq import aa_random
from plaster.run.prep import prep_worker
from plaster.tools.log.log import debug


def result_simple_fixture(has_decoy=False):
    """
    Generate a simple test fixture with some proteins and peptides
    """

    prep_params = PrepParams(
        proteins=[
            Munch(name="pro1", sequence="ABCDE"),
            Munch(name="pro2", sequence="BGBIJK BBLMN"),
        ]
    )

    pros = pd.DataFrame(
        dict(
            pro_id=["nul", "pro1", "pro2"],
            pro_is_decoy=[False, False, has_decoy],
            pro_i=[0, 1, 2],
            pro_ptm_locs=[None, None, None],
            pro_report=[None, None, None],
        )
    )

    # fmt: off
    pro_seqs = pd.DataFrame(
        dict(
            pro_i=[
                0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
            ],
            aa=[
                ".",
                "A", "B", "C", "D", "E",
                "B", "G", "B", "I", "J", "K", "B", "B", "L", "M", "N",
            ],
        )
    )
    # fmt: on

    peps = pd.DataFrame(
        dict(
            pep_i=[0, 1, 2, 3],
            pep_start=[0, 1, 6, 12],
            pep_stop=[1, 6, 12, 17],
            pro_i=[0, 1, 2, 2],
        )
    )

    # fmt: off
    pep_seqs = pd.DataFrame(
        dict(
            pep_i=[
                0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3
            ],
            aa=[
                ".",
                "A", "B", "C", "D", "E",
                "B", "G", "B", "I", "J", "K",
                "B", "B", "L", "M", "N",
            ],
            pep_offset_in_pro=[
                0,
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            ],
        )
    )
    # fmt: on

    return PrepResult(
        params=prep_params,
        _pros=pros,
        _pro_seqs=pro_seqs,
        _peps=peps,
        _pep_seqs=pep_seqs,
    )


def result_random_fixture(n_proteins):
    """
    Generate a fixture with randomly generate n_proteins
    """

    prep_params = PrepParams(
        proteins=[
            Munch(name=f"pro{i + 1}", sequence=aa_random(random.randrange(5, 100)))
            for i in range(n_proteins)
        ]
    )

    pro_spec_df = pd.DataFrame(
        dict(
            name=[f"pro{i + 1}" for i in range(n_proteins)],
            sequence=[aa_random(random.randrange(5, 100)) for i in range(n_proteins)],
            ptm_locs=[""] * n_proteins,
            report=[1] * n_proteins,
        )
    )

    pros_df, pro_seqs_df = prep_worker._step_2_create_pros_and_pro_seqs_dfs(pro_spec_df)

    peps_df, pep_seqs_df = prep_worker._step_4_proteolysis(pro_seqs_df, "trypsin")

    return PrepResult(
        params=prep_params,
        _pros=pros_df,
        _pro_seqs=pro_seqs_df,
        _peps=peps_df,
        _pep_seqs=pep_seqs_df,
    )


def result_4_count_fixture():
    prep_params = PrepParams(proteins=[Munch(name="pro1", sequence=".A.A.A.A"),])

    pros = pd.DataFrame(
        dict(
            pro_id=["nul", "four_count"],
            pro_is_decoy=[False, False],
            pro_i=[0, 1],
            pro_ptm_locs=[None, None],
            pro_report=[None, None],
        )
    )

    # fmt: off
    pro_seqs = pd.DataFrame(
        dict(
            pro_i=[
                0,
                1, 1, 1, 1, 1, 1, 1, 1,
            ],
            aa=[
                ".",
                ".", "A", ".", "A", ".", "A", ".", "A",
            ],
        )
    )
    # fmt: on

    peps = pd.DataFrame(
        dict(pep_i=[0, 1], pep_start=[0, 0], pep_stop=[1, 8], pro_i=[0, 1],)
    )

    # fmt: off
    pep_seqs = pd.DataFrame(
        dict(
            pep_i=[
                0,
                1, 1, 1, 1, 1, 1, 1, 1,
            ],
            aa=[
                ".",
                ".", "A", ".", "A", ".", "A", ".", "A",
            ],
            pep_offset_in_pro=[
                0,
                0, 1, 2, 3, 4, 5, 6, 7,
            ],
        )
    )
    # fmt: on

    return PrepResult(
        params=prep_params,
        _pros=pros,
        _pro_seqs=pro_seqs,
        _peps=peps,
        _pep_seqs=pep_seqs,
    )
