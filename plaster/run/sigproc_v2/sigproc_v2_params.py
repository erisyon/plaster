import re

from plaster.tools.calibration.calibration import Calibration
from plaster.tools.log.log import debug
from plaster.tools.schema.schema import Params, SchemaValidationFailed
from plaster.tools.schema.schema import Schema as s
from plaster.tools.log import log
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plaster.tools.utils import utils
from plumbum import local


class SigprocV2Params(Params):
    """
    About Calibration:
        The long term goal of the calibration files is to dissociate
        the name of the file from the records (subjects) in the file.
        For now, we're going to load all records from the calibration file
    """

    defaults = dict(
        divs=5,
        peak_mea=11,
        sig_limit=20.0,  # Keep peaks this many times brighter than the calibration background
        snr_thresh=2.0,  # Keep peaks with SNR greater than this
        n_dst_zslices=9,
        n_src_zslices=15,
        n_fields_limit=None,
        skip_anomaly_detection=True,
        skip_regional_balance=False,
    )

    schema = s(
        s.is_kws_r(
            calibration_file=s.is_str(),
            mode=s.is_str(options=common.SIGPROC_V2_MODES),
            divs=s.is_int(),
            peak_mea=s.is_int(),
            sig_limit=s.is_float(),
            snr_thresh=s.is_float(),
            n_dst_zslices=s.is_int(),
            n_src_zslices=s.is_int(),
            n_fields_limit=s.is_int(noneable=True),
            skip_anomaly_detection=s.is_bool(),
            skip_regional_balance=s.is_bool(),
        )
    )

    def validate(self):
        # Note: does not call super because the override_nones is set to false here
        self.schema.apply_defaults(self.defaults, apply_to=self, override_nones=False)
        self.schema.validate(self, context=self.__class__.__name__)

        if self.mode == common.SIGPROC_V2_PSF_CALIB:
            if local.path(self.calibration_file).exists():
                if not log.confirm_yn(
                    f"\nCalibration file '{self.calibration_file}' already exists "
                    "when creating a SIGPROC_V2_PSF_CALIB. Overwrite?",
                    "y",
                ):
                    raise SchemaValidationFailed(
                        f"Not overwriting calibration file '{self.calibration_file}'"
                    )

        else:
            if self.calibration_file != "":
                self.calibration = Calibration.load(self.calibration_file)

        return True


'''
    def set_radiometry_channels_from_input_channels_if_needed(self, n_channels):
        assert n_channels == 1
        self.radiometry_channels = {f"ch_0": 0}
        # if self.radiometry_channels is None:
        #     # Assume channels from nd2 manifest
        #     channels = list(range(n_channels))
        #     self.radiometry_channels = {f"ch_{ch}": ch for ch in channels}

    @property
    def n_output_channels(self):
        return 1
        # return len(self.radiometry_channels.keys())

    @property
    def n_input_channels(self):
        return 1
        # return len(self.radiometry_channels.keys())

    # @property
    # def channels_cycles_dim(self):
    #     # This is a cache set in sigproc_v1.
    #     # It is a helper for the repetitive call:
    #     # n_outchannels, n_inchannels, n_cycles, dim =
    #     return self._outchannels_inchannels_cycles_dim

    def _input_channels(self):
        """
        Return a list that converts channel number of the output to the channel of the input
        Example:
            input might have channels ["foo", "bar"]
            the radiometry_channels has: {"bar": 0}]
            Thus this function returns [1] because the 0th output channel is mapped
            to the "1" input channel
        """
        return [
            self.radiometry_channels[name]
            for name in sorted(self.radiometry_channels.keys())
        ]

    # def input_names(self):
    #     return sorted(self.radiometry_channels.keys())

    def output_channel_to_input_channel(self, out_ch):
        # return self._input_channels()[out_ch]
        return 0

    def input_channel_to_output_channel(self, in_ch):
        """Not every input channel necessarily has an output; can return None"""
        # return utils.filt_first_arg(self._input_channels(), lambda x: x == in_ch)
        return 0
'''
