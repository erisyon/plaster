import sys
from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_worker import ims_import
from plaster.tools.log.log import debug


class ImsImportTask(PipelineTask):
    def start(self):
        ims_import_params = ImsImportParams(**self.config.parameters)

        ims_import_result = ims_import(
            self.inputs.src_dir,
            ims_import_params,
            progress=self.progress,
            pipeline=self,
        )

        ims_import_result.save()
