from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.run.nn_v2.nn_v2_worker import nn_v2
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sigproc_v1.sigproc_v1_result import SigprocV1Result


class NNV2Task(PipelineTask):
    def start(self):
        nn_v2_params = NNV2Params(**self.config.parameters)

        sigproc_result = None
        if nn_v2_params.include_sigproc:
            sigproc_result = SigprocV1Result.load_from_folder(
                self.inputs.sigproc, prop_list=["n_cycles", "n_channels"]
            )

        prep_result = PrepResult.load_from_folder(self.inputs.prep)
        sim_v2_result = SimV2Result.load_from_folder(self.inputs.sim_v2)

        nn_v2_result = nn_v2(nn_v2_params, prep_result, sim_v2_result, sigproc_result, progress=self.progress, pipeline=self)

        nn_v2_result.save()
