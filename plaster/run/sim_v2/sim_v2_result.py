import pandas as pd
import numpy as np
from plaster.tools.utils import utils
from plaster.run.base_result import BaseResult, ArrayResult
from plaster.run.sim_v2.sim_v2_params import SimV2Params

# TODO: Move to a separate common module
DyeType = np.uint8
DyeWeightType = np.float32
RadType = np.float32
IndexType = np.uint32
RecallType = np.float32
ScoreType = np.float32
RowKType = np.float32


class SimV2Result(BaseResult):
    name = "sim_v2"
    filename = "sim_v2.pkl"

    required_props = dict(
        params=SimV2Params,
        train_dyemat=np.ndarray,  # unique (n_rows, n_channels * n_cycles)
        train_pep_recalls=np.ndarray,
        train_dyepeps=np.ndarray,  # (n, 3) where 3 are: (dyt_i, pep_i, count)
        train_radmat=(type(None), np.ndarray),
        train_true_pep_iz=(type(None), np.ndarray),
        train_true_dye_iz=(type(None), np.ndarray),
        train_true_row_ks=(type(None), np.ndarray),
        test_radmat=(type(None), np.ndarray),
        test_true_dye_iz=(type(None), np.ndarray),  # For debugging
        test_true_pep_iz=(type(None), np.ndarray),
        test_true_row_ks=(type(None), np.ndarray),
        _flus=(type(None), pd.DataFrame),  # Generated by this module
    )

    def __repr__(self):
        try:
            return (
                f"SimV2Result with {self.train_dyemat.shape[0]} training rows "
                f"and {self.test_dyemat.shape[0]} testing rows; with {self.train_dyemat.shape[1]} features"
            )
        except Exception:
            return "SimV2Result"

    def _generate_flu_info(self, prep_results):
        """
        Generates fluoro-sequence string like: "..0.1..1. ;1,2
        and adds in various counting statistics.  Note that the "head" portion
        of the flu is exactly n_edmans long, since edmans are the only kind of
        cycles that reveal a dye location.
        """

        def to_flu(x):
            n_channels = self.params.n_channels
            n_edmans = self.params.n_edmans
            full = utils.pad_list(
                list(x.aa), n_edmans
            )  # padded to head minimum but might be longer
            head = full[0:n_edmans]
            tail = full[n_edmans:]

            ch_to_n_head = [0] * n_channels
            ch_to_n_tail = [0] * n_channels
            for ch in range(n_channels):
                ch_to_n_head[ch] = sum(
                    [1 if self.params.ch_by_aa.get(aa, -1) == ch else 0 for aa in head]
                )
                ch_to_n_tail[ch] = sum(
                    [1 if self.params.ch_by_aa.get(aa, -1) == ch else 0 for aa in tail]
                )

            n_dyes_max_any_ch = np.max(np.array(ch_to_n_head) + np.array(ch_to_n_tail))

            flustr = (
                "".join([str(self.params.ch_by_aa.get(aa, ".")) for aa in head])
                + " ;"
                + ",".join([str(ch_to_n_tail[ch]) for ch in range(n_channels)])
            )

            ch_to_n_head_col_names = [f"n_head_ch_{ch}" for ch in range(n_channels)]
            ch_to_n_tail_col_names = [f"n_tail_ch_{ch}" for ch in range(n_channels)]

            df = pd.DataFrame(
                [
                    (
                        flustr,
                        *ch_to_n_head,
                        *ch_to_n_tail,
                        sum(ch_to_n_head),
                        sum(ch_to_n_tail),
                        sum(ch_to_n_head + ch_to_n_tail),
                        n_dyes_max_any_ch,
                    )
                ],
                columns=[
                    "flustr",
                    *ch_to_n_head_col_names,
                    *ch_to_n_tail_col_names,
                    "n_head_all_ch",
                    "n_tail_all_ch",
                    "n_dyes_all_ch",
                    "n_dyes_max_any_ch",
                ],
            )
            return df

        df = (
            prep_results._pep_seqs.groupby("pep_i")
            .apply(to_flu)
            .reset_index()
            .drop(["level_1"], axis=1)
            .sort_values("pep_i")
        )

        df_flu_count = df.groupby("flustr").size().reset_index(name="flu_count")
        self._flus = (
            df.set_index("flustr").join(df_flu_count.set_index("flustr")).reset_index()
        )

    def flat_train_radmat(self):
        assert self.train_radmat.ndim == 3
        return utils.mat_flatter(self.train_radmat)

    def flat_test_radmat(self):
        assert self.test_radmat.ndim == 3
        return utils.mat_flatter(self.test_radmat)

    def flat_train_dyemat(self):
        assert self.train_dyemat.ndim == 2
        return self.train_dyemat

    # def train_true_pep_iz(self):
    #     assert self.train_dyemat.ndim == 2
    #     shape = self.train_dyemat.shape
    #     return np.repeat(np.arange(shape[0]).astype(IndexType), shape[1])
    #
    # def test_true_pep_iz(self):
    #     shape = self.test_dyemat.shape
    #     return np.repeat(np.arange(shape[0]).astype(IndexType), shape[1])

    def flus(self):
        return self._flus

    def peps__flus(self, prep_result):
        return (
            prep_result.peps()
            .set_index("pep_i")
            .join(self._flus.set_index("pep_i"))
            .sort_index()
            .reset_index()
        )

    def peps__flus__unique_flus(self, prep_result):
        df = self.peps__flus(prep_result)
        return df[df.flu_count == 1]

    def pros__peps__pepstrs__flus(self, prep_result):
        return (
            prep_result.pros__peps__pepstrs()
            .set_index("pep_i")
            .join(self._flus.set_index("pep_i"))
            .sort_index()
            .reset_index()
        )

    def dump_debug(self):
        """
        Save properties out as numpy arrays for easier export
        """

        props = (
            "train_dyemat",
            "train_pep_recalls",
            "train_dyepeps",
            "train_radmat",
            "train_true_pep_iz",
            "train_true_dye_iz",
            "test_radmat",
            "test_true_dye_iz",
            "test_true_pep_iz",
        )
        for prop in props:
            np.save(f"_{prop}.npy", getattr(self, prop))
