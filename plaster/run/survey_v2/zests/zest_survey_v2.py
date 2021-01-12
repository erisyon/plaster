import numpy as np
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.survey_v2 import survey_v2_worker
from plaster.run.survey_v2.c import survey_v2 as survey_v2_fast
from plaster.run.survey_v2.survey_v2_params import SurveyV2Params
from plaster.tools.log.log import debug
from plaster.tools.utils import tmp
from zest import zest


@zest.skip(reason="broken")
def zest_survey_v2_pyx():
    # TODO: This

    prep_result = prep_fixtures.result_simple_fixture(True)

    sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(prep_result, labels="A,B")

    # pep 0:
    # pep 1: 10000 11000
    # pep 2: 00000 21100
    # pep 3: 00000 21000

    pep_i_to_mic_pep_i, pep_i_to_isolation_metric = survey_v2_fast.survey(
        prep_result.n_peps,
        sim_v2_result.train_dyemat,
        sim_v2_result.train_dyepeps,
        n_threads=1,
        progress=None,
    )

    assert pep_i_to_mic_pep_i.tolist() == [0, 3, 3, 2]

    # In the current verion they are all close
    # The first peptide should be a long way away and the other two should collide
    assert pep_i_to_isolation_metric[1] < 2
    assert pep_i_to_isolation_metric[2] < 2
    assert pep_i_to_isolation_metric[3] < 2

    # TODO: I really want to do a better job where I compare some contrived
    # peptides and make sure that the outliers are outliers
    # I also need to do the sampling to figure out that magic number of the "nothing close"

    # TODO: Test for unlabelled peptides. I'm sure it is broken

    zest()


@zest.skip(reason="Not complete")
def zest_survey_v2_integration():
    """
    This needs a lot of work on figuring out what the metric
    of success of the survey is exactly.

    Also need some brain-dead simpler cases.  Cases where
    the peptides are super clearly separated and make sure
    that we get sensible results.
    """

    with tmp.tmp_folder(chdir=True):
        prep_result = prep_fixtures.result_random_fixture(20)
        sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(
            prep_result, labels="DE,C,Y"
        )
        sim_v2_result.save()
        survey_v2_result = survey_v2_worker.survey_v2(
            SurveyV2Params(), prep_result, sim_v2_result
        )
        # survey_v2_result._survey.to_csv("/erisyon/internal/test.csv")

        # I will need to set the RNG on this to test.
        # There's a weird effect
        # https://docs.google.com/spreadsheets/d/1SrOjdNTpw7uLWU1iS7PFm4kbfNLTnW6Am2t85b-GKww/edit#gid=1462476311
        # Why are 3 peptides with the same flu not all showing each other as the nn?

    zest()
