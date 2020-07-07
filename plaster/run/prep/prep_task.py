import pandas as pd
from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.prep.prep_params import PrepParams
from plaster.run.prep.prep_worker import prep

from plaster.tools.log.log import debug


class PrepTask(PipelineTask):
    def start(self):
        prep_params = PrepParams(**self.config.parameters)
        pro_spec_df = pd.DataFrame(self.config.parameters.proteins)

        prep_result = prep(prep_params, pro_spec_df)

        prep_result.save()
