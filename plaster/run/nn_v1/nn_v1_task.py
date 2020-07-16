from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.sim_v1.sim_v1_result import SimV1Result
from plaster.run.prep.prep_result import PrepResult
from plaster.run.nn_v1.nn_v1_params import NNV1Params
from plaster.run.nn_v1.nn_v1_worker import nn_v1

# from plaster.tools.log.log import prof


class NNV1Task(PipelineTask):
    def start(self):
        nn_v1_params = NNV1Params(**self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_v1_result = SimV1Result.load_from_folder(self.inputs.sim)

        nn_v1_result = nn_v1(
            nn_v1_params,
            prep_result,
            sim_v1_result,
            progress=self.progress,
            pipeline=self,
        )

        nn_v1_result.save()
