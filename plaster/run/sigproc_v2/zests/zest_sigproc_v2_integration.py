from munch import Munch
import numpy as np
from math import ceil
from plumbum import local

from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2 import synth
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.image import imops
from plaster.tools.utils.utils import np_within
from plaster.tools.schema import check
from plaster.run.base_result import BaseResult, ArrayResult
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.tools.utils.fancy_indexer import FancyIndexer
from plaster.tools.log.log import debug

from zest import zest

@zest.skip(reason="coming soon")
def zest_sigproc_v2_integration():
    """
    Test Calibration and Analysis with synthethic data
    """

    raise NotImplementedError
