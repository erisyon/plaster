from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v1.sim_v1_params import SimV1Params
from plaster.run.sim_v1.sim_v1_worker import sim_v1
from plaster.tools.log.log import debug


class SimV1Task(PipelineTask):
    def start(self):
        sim_params = SimV1Params(include_dfs=True, **self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)

        sim_result = sim_v1(sim_params, prep_result, progress=self.progress, pipeline=self)
        sim_result._generate_flu_info(prep_result)
        sim_result.save()
