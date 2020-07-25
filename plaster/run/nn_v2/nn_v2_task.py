from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_worker
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.tools.pipeline.pipeline import PipelineTask


class TestNNTask(PipelineTask):
    def start(self):
        nn_v2_params = NNV2Params(**self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_v2_result = SimV2Result.load_from_folder(self.inputs.sim_v2)

        nn_v2_result = nn_worker(nn_v2_params, prep_result, sim_v2_result)
        nn_v2_result.save()
