from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.test_rf.test_rf_params import TestRFParams
from plaster.run.test_rf.test_rf_worker import test_rf
from plaster.run.sim_v1.sim_v1_result import SimV1Result
from plaster.run.train_rf.train_rf_result import TrainRFResult
from plaster.run.prep.prep_result import PrepResult


class TestRFTask(PipelineTask):
    def start(self):
        test_rf_params = TestRFParams(**self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_result = SimV1Result.load_from_folder(self.inputs.sim_v2)
        train_rf_result = TrainRFResult.load_from_folder(self.inputs.train_rf)

        test_rf_result = test_rf(
            test_rf_params,
            prep_result,
            sim_result,
            train_rf_result,
            progress=self.progress,
            pipeline=self,
        )

        test_rf_result.save()
