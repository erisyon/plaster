import numpy as np
from munch import Munch
from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.sigproc_v1.sigproc_v1_params import SigprocV1Params
from plaster.run.sigproc_v1.sigproc_v1_worker import sigproc
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.tools.log.log import debug


class SigprocV1Task(PipelineTask):
    def start(self):
        sigproc_params = SigprocV1Params(**self.config.parameters)

        ims_import_result = ImsImportResult.load_from_folder(self.inputs.ims_import)

        sigproc_params.set_radiometry_channels_from_input_channels_if_needed(
            ims_import_result.n_channels
        )

        sigproc_result = sigproc(sigproc_params, ims_import_result, self.progress)

        sigproc_result.save(
            save_full_signal_radmat_npy=sigproc_params.save_full_signal_radmat_npy
        )
