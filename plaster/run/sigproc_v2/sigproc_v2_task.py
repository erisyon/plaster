import numpy as np
from munch import Munch
from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plaster.tools.log.log import debug


class SigprocV2Task(PipelineTask):
    def start(self):
        sigproc_v2_params = SigprocV2Params(**self.config.parameters)

        ims_import_result = ImsImportResult.load_from_folder(self.inputs.ims_import)

        # sigproc_v2_params.set_radiometry_channels_from_input_channels_if_needed(
        #     ims_import_result.n_channels
        # )

        if sigproc_v2_params.mode in (common.SIGPROC_V2_PSF_CALIB, common.SIGPROC_V2_PSF_CALIB):
            sigproc_v2_instrument_calib_result = worker.sigproc_instrument_calib(
                sigproc_v2_params, ims_import_result, self.progress
            )
            sigproc_v2_instrument_calib_result.save()
            sigproc_v2_instrument_calib_result.calib.save(sigproc_v2_params.calibration_file)

        elif sigproc_v2_params.mode == common.SIGPROC_V2_INSTRUMENT_ANALYZE:
            sigproc_v2_result = worker.sigproc_analyze(
                sigproc_v2_params, ims_import_result, self.progress
            )
            sigproc_v2_result.save()

        else:
            raise ValueError("Unknown sigproc_v2 mode")
