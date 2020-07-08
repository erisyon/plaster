from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.run.sim_v2.sim_v2_worker import sim
from plaster.tools.log.log import debug


class SimV2Task(PipelineTask):
    def start(self):
        sim_v2_params = SimV2Params(include_dfs=True, **self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)

        sim_result_v2 = sim(
            sim_v2_params, prep_result, progress=self.progress, pipeline=self
        )
        sim_result_v2._generate_flu_info(prep_result)
        sim_result_v2.save()
