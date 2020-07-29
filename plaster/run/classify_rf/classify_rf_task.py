from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.classify_rf.classify_rf_params import ClassifyRFParams
from plaster.run.classify_rf.classify_rf_worker import classify_rf
from plaster.run.sim_v1.sim_v1_result import SimV1Result
from plaster.run.sigproc_v1.sigproc_v1_result import SigprocV1Result
from plaster.run.train_rf.train_rf_result import TrainRFResult


class ClassifyRFTask(PipelineTask):
    def start(self):
        classify_rf_params = ClassifyRFParams(**self.config.parameters)

        train_rf_result = TrainRFResult.load_from_folder(self.inputs.train_rf)

        sigproc_result = SigprocV1Result.load_from_folder(
            self.inputs.sigproc_v1, prop_list=["n_cycles", "n_channels"]
        )

        sim_result = SimV1Result.load_from_folder(
            self.inputs.sim_v1, prop_list=["params"]
        )

        classify_rf_result = classify_rf(
            classify_rf_params,
            train_rf_result,
            sigproc_result,
            sim_result.params,
            progress=self.progress,
        )

        classify_rf_result.save()
