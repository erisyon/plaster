"""
The error model contains every parameters relevant to calibration.

There are three kinds:
    * Chemistry related:
        p_edman_failure: The probability that Edman fails; creates a phase-shift in the data.
        p_detach: The probability that a molecule detaches from the surface; creates a sudden zero of signal

    * Dye related:
        gain: The brightness per dye in arbitrary camera units
        vpd: The extra variance that each dye induces

        beta: This is the older name for the gain parameter, now deprecated
        sigma: This is the older name for the variance parameter when it was in Log-Normal units, now deprecated

        p_bleach_per_cycle: The probability that an individual dye bleaches
        p_non_fluorescent: The probability that an individual dye is dud.
            Note that this is currently conflated with
            label.p_failure_to_bind_amino_acid and label.p_failure_to_attach_to_dye
            which are currently set to zero until we can disentangle them

    * Label related:
        p_failure_to_bind_amino_acid: Currently set to zero, see above
        p_failure_to_attach_to_dye: Currently set to zero, see above

"""

from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class ErrorModel(Params):
    schema = s(
        s.is_kws_r(
            p_dud=s.is_deprecated(),
            p_edman_failure=s.is_float(bounds=(0, 1)),
            p_detach=s.is_float(bounds=(0, 1)),
            dyes=s.is_list(
                elems=s.is_kws_r(
                    dye_name=s.is_str(),
                    p_bleach_per_cycle=s.is_float(bounds=(0, 1)),
                    p_non_fluorescent=s.is_float(bounds=(0, 1)),
                    # gain and vpd are the new parameters and beta, sigma are the legacy
                    gain=s.is_float(required=False, bounds=(0, None)),
                    vpd=s.is_float(required=False, bounds=(0, None)),
                    beta=s.is_float(required=False, bounds=(0, None)),
                    sigma=s.is_float(required=False, bounds=(0, None)),
                )
            ),
            labels=s.is_list(
                elems=s.is_kws_r(
                    label_name=s.is_str(),
                    p_failure_to_bind_amino_acid=s.is_float(bounds=(0, 1)),
                    p_failure_to_attach_to_dye=s.is_float(bounds=(0, 1)),
                )
            ),
        )
    )

    defaults = Munch(p_edman_failure=0.06, p_detach=0.05, dyes=[], labels=[])

    def __init__(self, **kwargs):
        dyes = kwargs["dyes"] = kwargs.pop("dyes", [])
        for dye in dyes:
            dye.p_bleach_per_cycle = dye.get(
                "p_bleach_per_cycle", kwargs.pop("p_bleach_per_cycle", 0.05)
            )
            dye.p_non_fluorescent = dye.get(
                "p_non_fluorescent", kwargs.pop("p_non_fluorescent", 0.07)
            )
        labels = kwargs["labels"] = kwargs.pop("labels", [])
        for label in labels:
            label.p_failure_to_bind_amino_acid = label.get(
                "p_failure_to_bind_amino_acid",
                kwargs.pop("p_failure_to_bind_amino_acid", 0.0),
            )
            label.p_failure_to_attach_to_dye = label.get(
                "p_failure_to_attach_to_dye",
                kwargs.pop("p_failure_to_attach_to_dye", 0.0),
            )
        super().__init__(**kwargs)

    @classmethod
    def no_errors(cls, n_channels, **kwargs):
        beta = kwargs.pop("beta", 7500.0)
        sigma = kwargs.pop("sigma", 0.0)
        gain = kwargs.pop("gain", 10.0)
        vpd = kwargs.pop("vpd", 0.1)
        return cls(
            p_edman_failure=0.0,
            p_detach=0.0,
            dyes=[
                Munch(
                    dye_name=f"dye_{ch}",
                    p_bleach_per_cycle=0.0,
                    p_non_fluorescent=0.0,
                    sigma=sigma,
                    beta=beta,
                    gain=gain,
                    vpd=vpd,
                )
                for ch in range(n_channels)
            ],
            labels=[
                Munch(
                    label_name=f"label_{ch}",
                    p_failure_to_bind_amino_acid=0.0,
                    p_failure_to_attach_to_dye=0.0,
                )
                for ch in range(n_channels)
            ],
            **kwargs,
        )

    @classmethod
    def from_err_set(cls, err_set, **kwargs):
        """err_set is a construct used by the error iterators in pgen"""
        n_channels = len(err_set.p_non_fluorescent)
        return cls(
            p_edman_failure=err_set.p_edman_failure[0],
            p_detach=err_set.p_detach[0],
            dyes=[
                Munch(
                    dye_name=f"dye_{ch}",
                    p_bleach_per_cycle=p_bleach_per_cycle,
                    p_non_fluorescent=p_non_fluorescent,
                    sigma=dye_sigma,
                    beta=dye_beta,
                    gain=dye_gain,
                    vpd=dye_vpd,
                )
                for ch, dye_beta, dye_sigma, dye_gain, dye_vpd, p_bleach_per_cycle, p_non_fluorescent in zip(
                    range(n_channels),
                    err_set.dye_beta,
                    err_set.dye_sigma,
                    err_set.dye_gain,
                    err_set.dye_vpd,
                    err_set.p_bleach_per_cycle,
                    err_set.p_non_fluorescent,
                )
            ],
            labels=[
                Munch(
                    label_name=f"label_{ch}",
                    p_failure_to_bind_amino_acid=0.0,
                    p_failure_to_attach_to_dye=0.0,
                )
                for ch in range(n_channels)
            ],
            **kwargs,
        )

    @classmethod
    def from_defaults(cls, n_channels):
        return cls(
            p_edman_failure=cls.defaults.p_edman_failure,
            p_detach=cls.defaults.p_detach,
            dyes=[
                Munch(
                    dye_name=f"dye_{ch}",
                    p_bleach_per_cycle=0.05,
                    p_non_fluorescent=0.07,
                    sigma=0.16,
                    beta=7500.0,
                    gain=7500.0,
                    vpd=0.10,
                )
                for ch in range(n_channels)
            ],
            labels=[
                Munch(
                    label_name=f"label_{ch}",
                    p_failure_to_bind_amino_acid=0.0,
                    p_failure_to_attach_to_dye=0.0,
                )
                for ch in range(n_channels)
            ],
        )

    def scale_dyes(self, key, scalar):
        for dye in self.dyes:
            dye[key] *= scalar

    def set_dye_param(self, key, val):
        for dye in self.dyes:
            dye[key] = val
