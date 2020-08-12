from zest import zest
import numpy as np
from plaster.run.prep import prep_fixtures
from plaster.run.sim_v2 import sim_v2_fixtures
from plaster.run.survey_v2.survey_v2_worker import dist_to_closest_neighbors, survey_v2
from plaster.run.survey_v2.survey_v2_params import SurveyV2Params
from plaster.tools.log.log import debug


def zest_dist_to_closest_neighbors():
    dyemat = np.array(
        [[0, 0, 0, 0, 0], [3, 3, 2, 1, 0], [2, 1, 1, 1, 1], [1, 1, 0, 0, 0],],
        dtype=np.uint8,
    )

    dyepeps = np.array(
        [
            # count, dye_i, pep_i
            [10, 1, 1],
            [5, 1, 2],
            [5, 2, 2],
        ],
        dtype=np.uint64,
    )

    def it_finds_metric():
        df = dist_to_closest_neighbors(dyemat, dyepeps)
        assert df.pep_i.to_list() == [1, 2]
        metric = df.collision_metric.to_list()
        assert metric[0] < metric[1]

    zest()


def zest_survey_v2():
    import pudb

    pudb.set_trace()
    prep_result = prep_fixtures.result_random_fixture(2)
    sim_v2_result = sim_v2_fixtures.result_from_prep_fixture(
        prep_result, n_labels=3, n_edmans=10
    )
    survey_v2_params = SurveyV2Params()
    survey_v2(survey_v2_params, prep_result, sim_v2_result)

    zest()
