from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.sim_v1.sim_v1_result import SimV1Result
from plaster.run.train_rf.train_rf_params import TrainRFParams
from plaster.run.train_rf.train_rf_worker import train_rf


class TrainRFTask(PipelineTask):
    def start(self):
        train_rf_params = TrainRFParams(**self.config.parameters)

        sim_result = SimV1Result.load_from_folder(self.inputs.sim_v2)

        train_rf_result = train_rf(train_rf_params, sim_result, progress=self.progress)

        train_rf_result.save()
