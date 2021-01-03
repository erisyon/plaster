import os
import numpy as np
from munch import Munch
from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plaster.tools.log.log import debug, important, colorful_exception


class SigprocV2Task(PipelineTask):
    def start(self):
        sigproc_v2_params = SigprocV2Params(**self.config.parameters)

        ims_import_result = ImsImportResult.load_from_folder(self.inputs.ims_import)

        if sigproc_v2_params.mode in (common.SIGPROC_V2_ILLUM_CALIB,):
            sigproc_v2_instrument_calib_result = worker.sigproc_instrument_calib(
                sigproc_v2_params, ims_import_result, self.progress
            )
            sigproc_v2_instrument_calib_result.save()

            # Save it to the current output in case it can not be written to the designated
            calib = sigproc_v2_instrument_calib_result.calib
            calib.set_identity(sigproc_v2_params.instrument_identity)
            calib.save_file("./calib.calib")

            # # SAVE to the designated, warn if failure
            # try:
            #     calib.save_file(
            #         sigproc_v2_params.calibration_file
            #     )
            # except Exception as e:
            #     # colorful_exception(e)
            #     important(f"Calib was not able to save to '{sigproc_v2_params.calibration_file}'. It was written to '{os.getcwd()}/calib.calib' as a backup.")

        elif sigproc_v2_params.mode == common.SIGPROC_V2_INSTRUMENT_ANALYZE:
            worker.sigproc_analyze(sigproc_v2_params, ims_import_result, self.progress)

        else:
            raise ValueError("Unknown sigproc_v2 mode")
