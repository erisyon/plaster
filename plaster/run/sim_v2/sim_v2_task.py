from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.run.sim_v2.sim_v2_worker import sim_v2
from plaster.tools.log.log import debug, prof


class SimV2Task(PipelineTask):
    def start(self):
        sim_v2_params = SimV2Params(include_dfs=True, **self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)

        sim_result_v2 = sim_v2(
            sim_v2_params, prep_result, progress=self.progress, pipeline=self
        )

        sim_result_v2.save()

        if sim_v2_params.dump_debug:
            sim_result_v2.dump_debug()
