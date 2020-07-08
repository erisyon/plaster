from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v1.sim_v1_result import SimV1Result
from plaster.run.survey_nn.survey_nn_params import SurveyNNParams
from plaster.run.survey_nn.survey_nn_worker import survey_nn


class SurveyNNTask(PipelineTask):
    def start(self):
        survey_nn_params = SurveyNNParams(**self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_result = SimV1Result.load_from_folder(self.inputs.sim)

        survey_nn_result = survey_nn(
            survey_nn_params,
            prep_result,
            sim_result,
            progress=self.progress,
            pipeline=self,
        )

        survey_nn_result.save()
