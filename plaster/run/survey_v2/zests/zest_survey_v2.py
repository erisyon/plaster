from zest import zest
import numpy as np
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.survey_v2.fast import survey_v2_fast
from plaster.run.survey_v2.survey_v2_params import SurveyV2Params
from plaster.tools.log.log import debug


def zest_survey_v2_pyx():
    prep_result = prep_fixtures.result_simple_fixture(2)

    sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(prep_result, n_labels=2)

    pep_i_to_isolation_metric = survey_v2_fast.survey(
        prep_result.n_peps,
        sim_v2_result.train_dyemat,
        sim_v2_result.train_dyepeps,
        n_threads=1,
        progress=None,
    )

    # The first peptide should be a long way away and the other two should collide
    assert pep_i_to_isolation_metric[1] > 5
    assert pep_i_to_isolation_metric[2] < 1
    assert pep_i_to_isolation_metric[3] < 1

    # TODO: I really want to do a better job where I compare some contrived
    # peptides and make sure that the outliers are outliers
    # I also need to do the sampling to figure out that magic number of the "nothing close"

    zest()
