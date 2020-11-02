import numpy as np
import pandas as pd
from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.aaseq.aaseq import aa_str_to_list
from plaster.run.error_model import ErrorModel
from plaster.tools.log.log import debug


class SimV1Params(Params):
    """
    Simulations parameters is and ErrorModel + parameters for sim
    """

    defaults = Munch(
        n_pres=1,
        n_mocks=0,
        n_edmans=1,
        n_samples_train=5_000,
        n_samples_test=1_000,
        dyes=[],
        labels=[],
        random_seed=None,
        train_n_sample_multiplier=None,  # This does not appear to be used anywhere. tfb
        allow_train_test_to_be_identical=False,
        enable_ptm_labels=False,
        is_survey=False,
    )

    schema = s(
        s.is_kws_r(
            is_survey=s.is_bool(),
            error_model=s.is_kws(**ErrorModel.schema.schema()),
            n_pres=s.is_int(bounds=(0, None)),
            n_mocks=s.is_int(bounds=(0, None)),
            n_edmans=s.is_int(bounds=(0, None)),
            n_samples_train=s.is_int(bounds=(1, None)),
            n_samples_test=s.is_int(bounds=(1, None)),
            dyes=s.is_list(
                elems=s.is_kws_r(dye_name=s.is_str(), channel_name=s.is_str())
            ),
            labels=s.is_list(
                elems=s.is_kws_r(
                    amino_acid=s.is_str(),
                    dye_name=s.is_str(),
                    label_name=s.is_str(),
                    ptm_only=s.is_bool(required=False, noneable=True),
                )
            ),
            random_seed=s.is_int(required=False, noneable=True),
            allow_train_test_to_be_identical=s.is_bool(required=False, noneable=True),
            enable_ptm_labels=s.is_bool(required=False, noneable=True),
        )
    )

    def copy(self):
        # REMOVE everything that _build_join_dfs put in
        utils.safe_del(self, "df")
        utils.safe_del(self, "by_channel")
        utils.safe_del(self, "ch_by_aa")

        dst = utils.munch_deep_copy(self, klass_set={SimV1Params})
        dst.error_model = ErrorModel(**dst.error_model)
        assert isinstance(dst, SimV1Params)
        return dst

    def __init__(self, include_dfs=True, **kwargs):
        kwargs["error_model"] = kwargs.pop("error_model", ErrorModel())
        super().__init__(**kwargs)
        if include_dfs:
            self._build_join_dfs()

    def validate(self):
        super().validate()

        all_dye_names = list(set([d.dye_name for d in self.dyes]))

        # No duplicate dye names
        self._validate(
            len(all_dye_names) == len(self.dyes), "The dye list contains a duplicate"
        )

        # No duplicate labels
        self._validate(
            len(list(set(utils.listi(self.labels, "amino_acid")))) == len(self.labels),
            "There is a duplicate label",
        )

        # All labels have a legit dye name
        [
            self._validate(
                label.dye_name in all_dye_names,
                f"Label {label.label_name} does not have a valid matching dye_name",
            )
            for label in self.labels
        ]

    @property
    def n_cycles(self):
        return self.n_pres + self.n_mocks + self.n_edmans

    def channels(self):
        return sorted(list(set(utils.listi(self.dyes, "channel_name"))))

    def channel_i_by_name(self):
        channels = self.channels()
        return {
            channel_name: channel_i for channel_i, channel_name in enumerate(channels)
        }

    @property
    def n_channels(self):
        return len(self.channel_i_by_name().keys())

    @property
    def n_channels_and_cycles(self):
        return self.n_channels, self.n_cycles

    def _build_join_dfs(self):
        """
        The error model contains information about the dyes and labels and other terms.
        Those error model parameters are wired together by names which are useful
        for reconciling calibrations.

        But here, these "by name" parameters are all put into a dataframe so that
        they can be indexed by integers.
        """
        sim_dyes_df = pd.DataFrame(self.dyes)
        assert len(sim_dyes_df) > 0

        sim_labels_df = pd.DataFrame(self.labels)
        assert len(sim_labels_df) > 0

        error_model_dyes_df = pd.DataFrame(self.error_model.dyes)
        assert len(error_model_dyes_df) > 0

        error_model_labels_df = pd.DataFrame(self.error_model.labels)
        assert len(error_model_labels_df) > 0

        if len(sim_dyes_df) > 0:
            channel_df = (
                sim_dyes_df[["channel_name"]]
                .drop_duplicates()
                .reset_index(drop=True)
                .rename_axis("ch_i")
                .reset_index()
            )

            label_df = pd.merge(
                left=sim_labels_df, right=error_model_labels_df, on="label_name"
            )

            dye_df = pd.merge(
                left=sim_dyes_df, right=error_model_dyes_df, on="dye_name"
            )
            dye_df = pd.merge(left=dye_df, right=channel_df, on="channel_name")

            self.df = (
                pd.merge(left=label_df, right=dye_df, on="dye_name")
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            self.df = pd.DataFrame()

        assert np.all(self.df.groupby("ch_i").p_bleach_per_cycle.nunique() == 1)
        assert np.all(self.df.groupby("ch_i").beta.nunique() == 1)
        assert np.all(self.df.groupby("ch_i").sigma.nunique() == 1)

        self.by_channel = [
            Munch(
                p_bleach_per_cycle=self.df[self.df.ch_i == ch]
                .iloc[0]
                .p_bleach_per_cycle,
                beta=self.df[self.df.ch_i == ch].iloc[0].beta,
                sigma=self.df[self.df.ch_i == ch].iloc[0].sigma,
            )
            for ch in range(self.n_channels)
        ]

        self.ch_by_aa = {row.amino_acid: row.ch_i for row in self.df.itertuples()}

    def to_label_list(self):
        """Summarize labels like: ["DE", "C"]"""
        return [
            "".join(
                [
                    label.amino_acid
                    for label in self.labels
                    if label.dye_name == dye.dye_name
                ]
            )
            for dye in self.dyes
        ]

    def to_label_str(self):
        """Summarize labels like: DE,C"""
        return ",".join(self.to_label_list())

    @classmethod
    def construct_from_aa_list(cls, aa_list, **kwargs):
        """
        This is a helper to generate channel when you have a list of aas.
        For example, two channels where ch0 is D&E and ch1 is Y.
        ["DE", "Y"].

        If you pass in an error model, it needs to match channels and labels.
        """

        check.list_or_tuple_t(aa_list, str)

        allowed_aa_mods = ["[", "]"]
        assert all(
            [
                (aa.isalpha() or aa in allowed_aa_mods)
                for aas in aa_list
                for aa in list(aas)
            ]
        )

        dyes = [
            Munch(dye_name=f"dye_{ch}", channel_name=f"ch_{ch}")
            for ch, _ in enumerate(aa_list)
        ]

        # Note the extra for loop because "DE" needs to be split into "D" & "E"
        # which is done by aa_str_to_list() - which also handles PTMs like S[p]
        labels = [
            Munch(
                amino_acid=aa,
                dye_name=f"dye_{ch}",
                label_name=f"label_{ch}",
                ptm_only=False,
            )
            for ch, aas in enumerate(aa_list)
            for aa in aa_str_to_list(aas)
        ]

        return cls(dyes=dyes, labels=labels, **kwargs)
