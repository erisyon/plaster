from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.sim.sim_result import SimResult
from plaster.run.train_rf.train_rf_params import TrainRFParams
from plaster.run.train_rf.train_rf_worker import train_rf


class TrainRFTask(PipelineTask):
    def start(self):
        train_rf_params = TrainRFParams(**self.config.parameters)

        sim_result = SimResult.load_from_folder(self.inputs.sim)

        train_rf_result = train_rf(train_rf_params, sim_result, progress=self.progress)

        train_rf_result.save()
