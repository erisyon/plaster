from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.sigproc_v1.sigproc_v1_result import SigprocV1Result
from plaster.run.calib_nn.calib_nn_params import CalibNNParams
from plaster.run.calib_nn.calib_nn_worker import calib_nn


class CalibNNTask(PipelineTask):
    def start(self):
        calib_nn_params = CalibNNParams(**self.config.parameters)

        sigproc_result = SigprocV1Result.load_from_folder(self.inputs.sigproc_v1)

        calib_nn_result = calib_nn(
            calib_nn_params, sigproc_result, progress=self.progress, pipeline=self,
        )

        calib_nn_result.save()
