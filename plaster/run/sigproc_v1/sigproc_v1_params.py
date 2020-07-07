import re
from plaster.tools.schema.schema import Schema as s, Params
from plaster.tools.utils import utils


class SigprocV1Params(Params):
    defaults = dict(
        hat_rad=2,
        iqr_rng=96,
        threshold_abs=1.0,
        channel_indices_for_alignment=None,
        channel_indices_for_peak_finding=None,
        radiometry_channels=None,
        save_debug=False,
        peak_find_n_cycles=4,
        peak_find_start=0,
        radial_filter=None,
        anomaly_iqr_cutoff=95,
        n_fields_limit=None,
        save_full_signal_radmat_npy=False,
    )

    schema = s(
        s.is_kws_r(
            anomaly_iqr_cutoff=s.is_number(noneable=True, bounds=(0, 100)),
            radial_filter=s.is_float(noneable=True, bounds=(0, 1)),
            peak_find_n_cycles=s.is_int(bounds=(1, None), noneable=True),
            peak_find_start=s.is_int(bounds=(0, None), noneable=True),
            save_debug=s.is_bool(),
            hat_rad=s.is_int(bounds=(1, 3)),
            iqr_rng=s.is_number(noneable=True, bounds=(0, 100)),
            threshold_abs=s.is_number(
                bounds=(0, 100)
            ),  # Not sure of a reasonable bound
            channel_indices_for_alignment=s.is_list(s.is_int(), noneable=True),
            channel_indices_for_peak_finding=s.is_list(s.is_int(), noneable=True),
            radiometry_channels=s.is_dict(noneable=True),
            n_fields_limit=s.is_int(noneable=True),
            save_full_signal_radmat_npy=s.is_bool(),
        )
    )

    def validate(self):
        # Note: does not call super because the override_nones is set to false here
        self.schema.apply_defaults(self.defaults, apply_to=self, override_nones=False)
        self.schema.validate(self, context=self.__class__.__name__)

        if self.radiometry_channels is not None:
            pat = re.compile(r"[0-9a-z_]+")
            for name, channel_i in self.radiometry_channels.items():
                self._validate(
                    pat.fullmatch(name),
                    "radiometry_channels name must be lower-case alphanumeric (including underscore)",
                )
                self._validate(
                    isinstance(channel_i, int), "channel_i must be an integer"
                )

    def set_radiometry_channels_from_input_channels_if_needed(self, n_channels):
        if self.radiometry_channels is None:
            # Assume channels from nd2 manifest
            channels = list(range(n_channels))
            self.radiometry_channels = {f"ch_{ch}": ch for ch in channels}

    @property
    def n_output_channels(self):
        return len(self.radiometry_channels.keys())

    @property
    def n_input_channels(self):
        return len(self.radiometry_channels.keys())

    @property
    def channels_cycles_dim(self):
        # This is a cache set in sigproc_v1.
        # It is a helper for the repeative call:
        # n_outchannels, n_inchannels, n_cycles, dim =
        return self._outchannels_inchannels_cycles_dim

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
        return self._input_channels()[out_ch]

    def input_channel_to_output_channel(self, in_ch):
        """Not every input channel necessarily has an output; can return None"""
        return utils.filt_first_arg(self._input_channels(), lambda x: x == in_ch)
