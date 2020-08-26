from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.run.survey_v2.survey_v2_params import SurveyV2Params
from plaster.run.survey_v2.survey_v2_worker import survey_v2


class SurveyV2Task(PipelineTask):
    def start(self):
        survey_v2_params = SurveyV2Params(**self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_result = SimV2Result.load_from_folder(self.inputs.sim_v2)

        survey_v2_result = survey_v2(
            survey_v2_params,
            prep_result,
            sim_result,
            progress=self.progress,
            pipeline=self,
        )

        survey_v2_result.save()
