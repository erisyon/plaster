from munch import Munch
import numpy as np
import pandas as pd
from zest import zest
from plaster.run.sigproc_v1 import sigproc_v1_worker
from plaster.run.sigproc_v1.sigproc_v1_result import SigprocV1Result
from plaster.tools.log.log import debug
from plaster.tools.utils.utils import npf, np_array_same
from plaster.tools.schema import check


@zest.skip("Not yet implemented")
def zest_find_anomalies():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_mask_anomalies():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_radiometry():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def _step_1_measure_quality():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def _step_2a_mask_anomalies():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def _step_2b_find_bg_median():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_step_2c_composite_channels():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_step_2_align():
    # field_df, ch_merged_cy_ims, raw_mask_rects = _step_2_align(raw_chcy_ims, sigproc_params)
    # check.df_t(field_df, SigprocResult.field_df_schema)
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_step_3_composite_aligned_images():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_step_4_find_peaks():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_step_5_radiometry():
    raise NotImplementedError


@zest.skip("Not yet implemented")
def zest_do_field():
    raise NotImplementedError
