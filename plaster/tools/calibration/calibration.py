import base64
import numpy as np
import re
from munch import Munch
from plaster.tools.utils import utils
from plumbum import local
from plaster.run.sigproc_v2.reg_psf import RegPSF
from plaster.tools.log.log import debug


"""
There are are variety of un-related calibration data-sets
    - PSF per instrument
    - Illumination per instrument
    - Bleaching rate of a specific dye
    - Dud-dye rate of a dye
    - Others I don't know about yet

The business logic doesn't typically care what instrument or name
of dye was use, it just wants values. But the high levlel code
always cares to sanity check that the calibration that is loaded matches
the data source in question.

I want to make sure that identity (serial number, etc) is never
semantically encoded in filename.. that is, such identiyt inforation should
be authoritative from metadata inside the file/record.

There would be a loader that would sanity check the identiy and then
stip that identity and pass the calibraiton data along to the
inner business logic that cares about it.

Example:
    RegPSF: (n_divs, n_divs, 3)
    RegIlluminationBalance: (n_divs, n_divs)

    Both of these are generates by a 1-count multi-dye expriment
    on a specific insrument using a specific calibrant peptide.
    Then they are re-used by every sigproc run from that instrument


plaster_main:
    lookup the instrument and dye serial numbers referenced in the run.yaml
    Calibation shoul dbe laded based on the iodentiy referenced in a data-source
    As the calib store for records associated and those identi
    Pass along identityless records to
    Separate the validation

    identity, data = load_intrument_raw_data
    calib.load_from_identity(identity)

Two things:
    - Load/find records based on identity
    - Validate that the records conform to versioned schema
        - Sanity check date (warn if calib is too old for example)
    - Conversion from some possibly serialized, (human readable?)
      form into an object that the business-logic code actually needs operate.


Ideas for refactor:

class CalibRecord:
    identity: CalibIdentity
    type: CalibType
    value: schemaless?


class CalibBag:
    '''A container of CalibRecords''''

    @classmethod
    def load_identity(cls, identity: CalibIdentity)

"""


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
        regional_psf=RegPSF,
        fg_mean=float,
        fg_std=float,
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
        re.compile("regional_psf\." + instrument_channel_pat),
        re.compile("zstack_depths\." + instrument_pat),
        re.compile("p_failure_to_bind_amino_acid\." + label_aa_pat),
        re.compile("p_failure_to_attach_to_dye\." + label_aa_pat),
        re.compile("fg_mean\." + instrument_channel_pat),
        re.compile("fg_std\." + instrument_channel_pat),
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
        (local.path(path) / "..").mkdir()
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

    def psfs(self, ch_i):
        # Backward compatability. To be deprecated
        old_key = f"regional_psf_zstack.instrument_channel[{ch_i}]"
        if old_key in self:
            old_psf = np.array(self[old_key])
            in_focus_ims = old_psf[old_psf.shape[0] // 2]
            im_mea = 512  # Hard coded for now
            return RegPSF.from_psf_ims(im_mea, in_focus_ims)

        reg_psf = self[f"regional_psf.instrument_channel[{ch_i}]"]
        return reg_psf

    def __init__(self, propsubs=None):
        super().__init__()
        self.add(propsubs)
