from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v1.sim_v1_result import SimV1Result
from plaster.run.survey_nn_v1.survey_nn_v1_params import SurveyNNV1Params
from plaster.run.survey_nn_v1.survey_nn_v1_worker import survey_nn_v1


class SurveyNNV1Task(PipelineTask):
    def start(self):
        survey_nn_v1_params = SurveyNNV1Params(**self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_result = SimV1Result.load_from_folder(self.inputs.sim_v2)

        survey_nn_v1_result = survey_nn_v1(
            survey_nn_v1_params,
            prep_result,
            sim_result,
            progress=self.progress,
            pipeline=self,
        )

        survey_nn_v1_result.save()
