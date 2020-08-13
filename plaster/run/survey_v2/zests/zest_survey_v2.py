from zest import zest
import numpy as np
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.survey_v2.fast import survey_v2_fast
from plaster.run.survey_v2.survey_v2_params import SurveyV2Params
from plaster.tools.log.log import debug


def zest_survey_v2_pyx():
    prep_result = prep_fixtures.result_simple_fixture(2)

    sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(prep_result)

    survey_v2_fast.survey(
        sim_v2_result.train_dyemat,
        sim_v2_result.train_dyepeps,
        n_threads=1,
        progress=None,
    )

    zest()
