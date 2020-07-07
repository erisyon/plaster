from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.sim.sim_result import SimResult
from plaster.run.prep.prep_result import PrepResult
from plaster.run.test_nn.test_nn_params import TestNNParams
from plaster.run.test_nn.test_nn_worker import test_nn

# from plaster.tools.log.log import prof


class TestNNTask(PipelineTask):
    def start(self):
        test_nn_params = TestNNParams(**self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_result = SimResult.load_from_folder(self.inputs.sim)

        test_nn_result = test_nn(
            test_nn_params,
            prep_result,
            sim_result,
            progress=self.progress,
            pipeline=self,
        )

        test_nn_result.save()
