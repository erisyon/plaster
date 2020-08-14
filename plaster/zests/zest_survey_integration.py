from plaster.tools.utils import utils
from plumbum import local, FG
from zest import zest


@zest.skip(reason="WIP")
def zest_survey_integration():
    """
    Show that a survey gen and run can execute
    """

    csv_file = "/tmp/__zest_survey_integration.csv"

    utils.save(
        csv_file,
        utils.smart_wrap(
            """
            Name,Seq,Abundance,POI
            pep0,ALNCLVMQL,1,1
            pep1,APHGVVFL,1,1
            pep2,KIADYNYML,1,1
            pep3,MLPDDFTGC,4,1
            pep4,CCQSLQTYV,1,1
            pep5,TLMSKTQSL,1,1
            pep6,VLCMNQKLI,1,1
            pep7,ACCDFTAKV,1,0
            """,
            assert_if_exceeds_width=True,
        ),
    )

    local["p"][
        "gen",
        "survey",
        "--sample=zest_survey_integration",
        f"--protein_csv={csv_file}",
        "--label_set=C,M",
        "--n_pres=1",
        "--n_mocks=0",
        "--n_edmans=15",
        "--force",
        "--job=./jobs_folder/__zest_survey_integration",
    ] & FG

    local["p"]["run", "./jobs_folder/__zest_survey_integration"] & FG

    zest()
