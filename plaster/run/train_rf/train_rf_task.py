from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.sim_v1.sim_v1_result import SimV1Result
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.run.train_rf.train_rf_params import TrainRFParams
from plaster.run.train_rf.train_rf_worker import train_rf


class TrainRFTask(PipelineTask):
    def start(self):
        train_rf_params = TrainRFParams(**self.config.parameters)

        if "sim_v1" in self.inputs:
            sim_result = SimV1Result.load_from_folder(self.inputs.sim_v1)
        elif "sim_v2" in self.inputs:
            sim_result = SimV2Result.load_from_folder(self.inputs.sim_v2)

        train_rf_result = train_rf(train_rf_params, sim_result, progress=self.progress)

        train_rf_result.save()
