import base64
import numpy as np
import re
from munch import Munch
from plaster.tools.utils import utils


class Calibration(Munch):
    """
    This is a key/value system with name and type checking for calibrations.
    It may become necessary at some point to have a proper
    database for this kind of information, this class is intended
    to validate these records making it easier to transition to
    a database at some point.

    Calibrant records have three fields:
        property, subject_type, subject_id

    Together this tuple is called a "propsub".

        * property
            The attribute or variable in question.
            Examples:
                regional_background
                regional_brightness_correction
                brightness__one_dye__mean
                brightness__one_dye__std
            (See Naming Guidelines below.)

        * subject_type
            Example:
                instrument
                label_cysteine

        * subject_id
            Examples:
                batch_2020_03_04
                serial_1234567

    propsubs are listed as key/value pairs like:

        property.subject_type.subject_id = 1

    When loaded, this class never assigns meaning to the path name;
    all important information is inside the yaml files.

    To simplify communicating with other systems, the loaded
    calibration can filter for subjects thus stripping subject_ids them
    from the propsubs. This allows library code to proceed
    without having to know the subject_id.

    For example, suppose the sim() function need a value
    for the "p_failure_to_bind_amino_acid.label_cysteine = 1".

    But, the actual record in calib is:
        "p_failure_to_bind_amino_acid.label_cysteine.batch_2020_03_16 = 1"
    which includes the subject_id (batch_2020_03_16)

    To prevent the sim() function from needing the subject_id,
    Calibration class can be filtered to create this:
        "label_cysteine.p_failure_to_bind_amino_acid = 1"

    Calibration objects can be merged together. Example:
        calib = Calibration.from_yaml("abbe_atto_647.yaml")
        calib.update(Calibration.from_yaml("chemistry_set_1.yaml"))
        calib.updte(Calibration.from_yaml("another.yaml"))

    To prevent proliferation of fields, all fields
    are also declared and validated.

    Naming guidelines:
        * All fields as symbols, ie: ([a-z][a-z_0-9]*), that is: lower_snake_case
            Good:
              brightness_247
            Bad:
              Brightness247
              247Brightness
        * Multi-part names go from less_specific -> more_specific
          Example:
              brightness_one_dye_atto_647
        * Variants are last (like "mean" or "std")
        * When a date is referenced, it should be in YYYY_MM_DD form.

    Example read usage:
        subjects_of_interest = ["instrument.serial_number_1234", "cysteine.batch_2020_03_16"]
        cal = Calibration.from_yaml("somefile.yml")
        cal.update(Calibration.from_yaml("anotherfile.yml"))

    Example write usage:
        instrument_id = "1234"
        c = Calibration({
            f"regional_background.instrument.{instrument_id}": value,
            f"metadata.instrument.{instrument_id}": dict(a=1, b=2)
        })
        c.to_yaml(path)
    """

    properties = dict(
        regional_illumination_balance=list,
        regional_fg_threshold=list,
        regional_bg_mean=list,
        regional_bg_std=list,
        regional_psf_zstack=list,
        zstack_depths=list,
        p_failure_to_bind_amino_acid=float,
        p_failure_to_attach_to_dye=float,
        metadata=dict,
    )

    symbol_pat = r"[a-z][a-z0-9_]*"
    instrument_pat = r"instrument"
    instrument_channel_pat = r"instrument_channel\[([0-9])\]"
    label_aa_pat = r"label\[([A-Z])\]"

    subject_type_patterns = [
        # (subj_type_pattern, allowed subj_id_pattern)
        (re.compile(instrument_pat), re.compile(symbol_pat)),
        (re.compile(instrument_channel_pat), re.compile(symbol_pat)),
        (re.compile(label_aa_pat), re.compile(symbol_pat)),
    ]

    propsub_pats = [
        re.compile("metadata\.instrument"),
        re.compile("regional_illumination_balance\." + instrument_channel_pat),
        re.compile("regional_fg_threshold\." + instrument_channel_pat),
        re.compile("regional_bg_mean\." + instrument_channel_pat),
        re.compile("regional_bg_std\." + instrument_channel_pat),
        re.compile("regional_psf_zstack\." + instrument_channel_pat),
        re.compile("zstack_depths\." + instrument_pat),
        re.compile("p_failure_to_bind_amino_acid\." + label_aa_pat),
        re.compile("p_failure_to_attach_to_dye\." + label_aa_pat),
    ]

    bracket_pat = re.compile(r"([^\[]+)\[([^\]]+)\]")

    def _split_key(self, key):
        parts = key.split(".")
        if len(parts) == 2:
            return (*parts, None)
        elif len(parts) == 3:
            return parts
        else:
            raise TypeError(f"key '{key}' not a valid calibration key")

    def validate(self):
        for key, val in self.items():
            prop, subj_type, subj_id = self._split_key(key)

            # VALIDATE subj_type
            found_subj_id_pat = None
            for subj_type_pat, subj_id_pat in self.subject_type_patterns:
                m = subj_type_pat.match(subj_type)
                if m is not None:
                    found_subj_id_pat = subj_id_pat
                    break
            else:
                raise TypeError(
                    f"subject_type '{subj_type}' is not a valid subject_type"
                )

            # VALIDATE subj_id if present
            if subj_id is not None:
                if not found_subj_id_pat.match(subj_id):
                    raise TypeError(
                        f"subject_id '{subj_id}' does not match pattern for subject_type '{subj_type}'"
                    )

            # VALIDATE property
            expecting_prop_type = self.properties.get(prop)
            if expecting_prop_type is None:
                raise TypeError(f"property '{prop}' was not found")
            if not isinstance(val, (expecting_prop_type,)):
                raise TypeError(
                    f"property '{prop}' was expecting val of type {expecting_prop_type} but got {type(val)}."
                )

            # VALIDATE property / subject_type
            prop_subj_type = f"{prop}.{subj_type}"
            for propsub_pat in self.propsub_pats:
                if propsub_pat.match(prop_subj_type) is not None:
                    break
            else:
                raise TypeError(f"'{prop_subj_type}' if not a valid calibration")

    def filter_subject_ids(self, subject_ids_to_keep):
        keep = {}
        for key, val in self.items():
            prop, subj_type, subj_id = self._split_key(key)
            if subj_id is not None and subj_id in subject_ids_to_keep:
                keep[f"{prop}.{subj_type}"] = val

        self.clear()
        self.update(keep)
        self.validate()
        return self

    def has_subject_ids(self):
        for key, val in self.items():
            prop, subj_type, subj_id = self._split_key(key)
            if subj_id is not None:
                return True
        return False

    def add(self, propsubs):
        if propsubs is not None:
            self.update(propsubs)
        self.validate()
        return self

    def save(self, path):
        utils.pickle_save(path, self)
        return self

    @classmethod
    def load(cls, path):
        return utils.pickle_load(path)

    def is_empty(self):
        return len(self.keys()) == 0

    def set_subject_id(self, subject_id):
        new_propsubs = {}
        for key, val in self.items():
            prop, subject_type, subj_id = self._split_key(key)
            assert subj_id is None
            subj_id = subject_id
            new_propsubs[".".join((prop, subject_type, subj_id))] = val
        self.clear()
        self.add(new_propsubs)

    def __init__(self, propsubs=None):
        super().__init__()
        self.add(propsubs)
