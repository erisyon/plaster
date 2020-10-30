"""
The error model contains every parameters relevant to calibration.

There are three kinds:
    * Chemistry related:
        p_edman_failure: The probability that Edman fails; creates a phase-shift in the data.
        p_detach: The probability that a molecule detaches from the surface; creates a sudden zero of signal

    * Dye related:
        beta: This is the mean of the log-normal
        sigma: This is std of a log-normal intensity in log space.
            ie intensity = norm(mu=np.log(beta * dye_count), sigma=sigma)
        zero_beta: This is the mean of the zero-count (dark)
        zero_sigma: This is the std of the zero-count intensities

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
from dataclasses import dataclass
from typing import List
from plaster.tools.schema.schema import Schema as s, Params


@dataclass
class ChGainModel:
    beta: float
    sigma: float
    zero_beta: float
    zero_sigma: float


@dataclass
class GainModel:
    row_k_beta: float
    row_k_sigma: float
    channels: List[ChGainModel]

    @property
    def n_channels(self):
        return len(self.channels)

    @classmethod
    def test_fixture(cls):
        return GainModel(
            row_k_beta=1.0,
            row_k_sigma=0.0,
            channels=[
                ChGainModel(beta=7000.0, sigma=0.20, zero_beta=0.0, zero_sigma=200.0)
            ],
        )


class ErrorModel(Params):
    schema = s(
        s.is_kws_r(
            row_k_beta=s.is_float(),
            row_k_sigma=s.is_float(),
            p_dud=s.is_deprecated(),
            p_edman_failure=s.is_float(bounds=(0, 1)),
            p_detach=s.is_float(bounds=(0, 1)),
            dyes=s.is_list(
                elems=s.is_kws_r(
                    dye_name=s.is_str(),
                    p_bleach_per_cycle=s.is_float(bounds=(0, 1)),
                    p_non_fluorescent=s.is_float(bounds=(0, 1)),
                    beta=s.is_float(required=False, bounds=(0, None)),
                    sigma=s.is_float(required=False, bounds=(0, None)),
                    zero_beta=s.is_float(required=False),
                    zero_sigma=s.is_float(required=False, bounds=(0, None)),
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

    defaults = Munch(
        row_k_beta=1.0, row_k_sigma=0.0, p_edman_failure=0.06, p_detach=0.05, dyes=[], labels=[]
    )

    def to_gain_model(self):
        return GainModel(
            row_k_beta=self.row_k_beta,
            row_k_sigma=self.row_k_sigma,
            channels=[
                ChGainModel(
                    beta=dye.beta,
                    sigma=dye.sigma,
                    zero_beta=dye.zero_beta,
                    zero_sigma=dye.zero_sigma,
                )
                for dye in self.dyes
            ],
        )

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
        zero_beta = kwargs.pop("zero_beta", 0.0)
        zero_sigma = kwargs.pop("zero_sigma", 0.0)
        p_bleach = kwargs.pop("p_bleach", 0.0)
        p_non_fluorescent = kwargs.pop("p_non_fluorescent", 0.0)
        return cls(
            p_edman_failure=0.0,
            p_detach=0.0,
            dyes=[
                Munch(
                    dye_name=f"dye_{ch}",
                    p_bleach_per_cycle=p_bleach,
                    p_non_fluorescent=p_non_fluorescent,
                    sigma=sigma,
                    beta=beta,
                    zero_beta=zero_beta,
                    zero_sigma=zero_sigma,
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
                    zero_beta=dye_zero_beta,
                    zero_sigma=dye_zero_sigma,
                )
                for ch, dye_beta, dye_sigma, dye_zero_beta, dye_zero_sigma, p_bleach_per_cycle, p_non_fluorescent in zip(
                    range(n_channels),
                    err_set.dye_beta,
                    err_set.dye_sigma,
                    err_set.zero_beta,
                    err_set.zero_sigma,
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
                    zero_beta=300.0,
                    zero_sigma=700.0,
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
