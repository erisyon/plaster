from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.run.prep.prep_result import PrepResult
from plaster.run.sim.sim_params import SimParams
from plaster.run.sim.sim_worker import sim
from plaster.tools.log.log import debug


class SimTask(PipelineTask):
    def start(self):
        sim_params = SimParams(include_dfs=True, **self.config.parameters)

        prep_result = PrepResult.load_from_folder(self.inputs.prep)

        sim_result = sim(sim_params, prep_result, progress=self.progress, pipeline=self)
        sim_result._generate_flu_info(prep_result)
        sim_result.save()
