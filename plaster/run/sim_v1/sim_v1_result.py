import numpy as np
import pandas as pd
from plaster.run.base_result import ArrayResult, BaseResult
from plaster.run.sim_v1.sim_v1_params import SimV1Params
from plaster.tools.utils import utils

DyeType = np.uint8
DyeWeightType = np.float32
RadType = np.float32
IndexType = np.uint32
RecallType = np.float32
ScoreType = np.float32


class SimV1Result(BaseResult):
    name = "sim_v1"
    filename = "sim_v1.pkl"

    required_props = dict(
        params=SimV1Params,
        train_true_pep_iz=np.ndarray,
        train_dyemat=ArrayResult,  # (n_peps, n_samples, n_channels, n_cycles):uint8
        train_radmat=ArrayResult,  # (n_peps, n_samples, n_channels, n_cycles):float32
        train_recalls=ArrayResult,  # (n_peps):float32
        train_flus=np.ndarray,
        train_flu_remainders=np.ndarray,
        test_true_pep_iz=(type(None), np.ndarray),
        test_dyemat=(type(None), ArrayResult),
        test_radmat=(type(None), ArrayResult),
        test_flus=(type(None), np.ndarray),
        test_flu_remainders=(type(None), np.ndarray),
        test_recalls=(type(None), ArrayResult),
        _flus=pd.DataFrame,
    )

    def __repr__(self):
        try:
            return (
                f"SimV1Result with {self.train_dyemat.shape[0]} training rows "
                f"and {self.test_dyemat.shape[0]} testing rows; with {self.train_dyemat.shape[1]} features"
            )
        except Exception:
            return "SimV1Result"

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
        shape = self.train_dyemat.shape
        return self.train_radmat.reshape((shape[0], shape[1] * shape[2]))

    def flat_test_radmat(self):
        shape = self.test_radmat.shape
        return self.test_radmat.reshape((shape[0], shape[1] * shape[2]))

    # def train_true_pep_iz(self):
    #     shape = self.train_dyemat.shape
    #     return np.repeat(np.arange(shape[0]).astype(IndexType), shape[1])
    #
    # def test_true_pep_iz(self):
    #     shape = self.test_dyemat.shape
    #     return np.repeat(np.arange(shape[0]).astype(IndexType), shape[1])

    # def unflat(self, mat_name):
    #     """
    #     Used to extract an un-flattened form of a mat (rows, channels, cycles)
    #     Example:
    #         run.sim.unflat("train_dyemat")  # returns a 3D mat instead of 2D
    #     """
    #     n_channels, n_cycles = self.params.n_channels_and_cycles
    #     mat = self.__getattr__(mat_name)
    #     assert mat.ndim == 2
    #     return utils.mat_lessflat(mat, n_channels, n_cycles)

    # def pep_i_to_flu_brightness(self, n_peps):
    #     """
    #     A helper to sum up the dye counts in all channels for each pep.
    #     """
    #
    #     # THIS is all messed up .... train_true_pep_iz
    #     # is per ROW. there are thoufsand of them
    #     # I ust want the flus to be per peptide bot the keepers!
    #
    #     return np.sum(
    #         ~np.isnan(self.test_flus), axis=1
    #     ) + np.sum(  # Number of labels (anything non-nan)
    #         self.test_flu_remainders, axis=1
    #     )  # Plus the remainders

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
