import random
from munch import Munch
import pandas as pd
import numpy as np
from plaster.run.base_result import BaseResult
from plaster.run.prep.prep_params import PrepParams
from plaster.tools.aaseq.aaseq import aa_random
from plaster.tools.utils import utils
from plaster.tools.log.log import debug


class PrepResult(BaseResult):
    """
    Follows the "Composite DataFrame Pattern"

    _pros DataFrame:
        pro_id, pro_is_decoy, pro_i, pro_ptm_locs, pro_report

    _pro_seqs DataFrame:
        pro_i, aa

    _peps DataFrame:
        pep_i, pep_start, pep_stop (positions in the parent protein), pro_i (parent protein)

    _pep_seqs DataFrame:
        pep_i, aa, pep_offset_in_pro

    """

    name = "prep"
    filename = "prep.pkl"

    pros_columns = ["pro_id", "pro_is_decoy", "pro_i", "pro_ptm_locs", "pro_report"]
    pro_seqs_columns = ["pro_i", "aa"]

    peps_columns = ["pep_i", "pep_start", "pep_stop", "pro_i"]
    pep_seqs_columns = ["pep_i", "aa", "pep_offset_in_pro"]

    required_props = dict(
        params=PrepParams,
        _pros=pd.DataFrame,
        _pro_seqs=pd.DataFrame,
        _peps=pd.DataFrame,
        _pep_seqs=pd.DataFrame,
    )

    def _none_abundance_to_nan(self, df):
        if "abundance" in df:
            df.abundance.fillna(value=np.nan, inplace=True)

    def pros(self):
        _pros = self._pros
        prep_pros_df = pd.DataFrame(self.params.proteins)
        if "abundance" in prep_pros_df:
            self._none_abundance_to_nan(prep_pros_df)
            _pros = _pros.set_index("pro_id").join(
                prep_pros_df.set_index("name")[["abundance"]]
            )
        return _pros.reset_index()

    @property
    def n_pros(self):
        return len(self._pros)

    @property
    def n_pros_of_interest(self):
        # How many proteins are considered "of interest" is recorded in the pros df as "pro_report"
        # This can be set via pgen --protein_of_interest or by calling the PrepResult routine
        # set_proteins_of_interest( list_of_protein_ids )
        n_in_report = len(self.pros__in_report())
        return n_in_report

    def set_pros_of_interest(self, protein_ids=[]):
        if type(protein_ids) is not list:
            protein_ids = [protein_ids]
        assert all([id in self._pros.pro_id.values for id in protein_ids])
        self._pros.pro_report = self._pros.apply(
            lambda x: 1 if x.pro_id in protein_ids else 0, axis=1
        )

    def set_pro_ptm_locs(self, protein_id="", ptms=""):
        assert type(protein_id) is str
        assert type(ptms) is str
        assert protein_id in self._pros.pro_id.values
        self._pros.loc[self._pros.pro_id == protein_id, "pro_ptm_locs"] = ptms

    def get_pro_ptm_locs(self, protein_id=""):
        return self._pros.loc[self._pros.pro_id == protein_id, "pro_ptm_locs"].iloc[0]

    def pros_abundance(self):
        df = self.pros()
        if "abundance" in df.columns:
            self._none_abundance_to_nan(df)
            return np.nan_to_num(df.abundance.values)
        return None

    def peps_abundance(self):
        df = self.pros__peps()
        if "abundance" in df:
            self._none_abundance_to_nan(df)
            return np.nan_to_num(df.abundance.values)
        return None

    def pros__in_report(self):
        df = (
            self.pros()
        )  # why not self._pros? because I want abundance if it is available.
        return df[df.pro_report > 0]

    def pros__ptm_locs(self):
        return self._pros[self._pros.pro_ptm_locs != ""]

    def pros__from_decoys(self):
        return self._pros[self._pros.pro_is_decoy > 0]

    def pros__no_decoys(self):
        return self._pros[self._pros.pro_is_decoy < 1]

    def proseqs(self):
        return self._pro_seqs

    def peps(self):
        return self._peps

    def pepseqs(self):
        return self._pep_seqs

    @property
    def n_peps(self):
        return len(self._peps)

    def pros__peps(self):
        return (
            self.pros()
            .set_index("pro_i")
            .join(self.peps().set_index("pro_i"), how="left")
            .reset_index()
        )

    def peps__no_decoys(self):
        return (
            self.pros__no_decoys()
            .set_index("pro_i")
            .join(self.peps().set_index("pro_i"), how="left")
            .reset_index()[self.peps_columns]
        )

    def peps__from_decoys(self):
        return (
            self.pros__from_decoys()
            .set_index("pro_i")
            .join(self.peps().set_index("pro_i"), how="left")
            .reset_index()[self.peps_columns]
        )

    def peps__in_report(self):
        return (
            self.pros__in_report()
            .set_index("pro_i")
            .join(self.peps().set_index("pro_i"), how="left")
            .reset_index()[self.peps_columns]
        )

    def peps__ptms(
        self,
        include_decoys=False,
        in_report_only=False,
        ptm_peps_only=True,
        ptms_to_rows=True,
    ):
        """
        Create a df that contains peptides and the ptm locations they contain.

        include_decoys: should peptides from decoy proteins be included?
        in_report_only: should only "proteins of interest" be included?
        ptm_peps_only : should only peptides that contain ptm locations be included?
        ptms_to_rows  : should ;-delimited pro_ptms_loc be 'unrolled'/nomralized into a ptm column?

        Returns a dataframe.
        """

        df = self.pros__peps()
        df = df[df.pro_i != 0]  # get rid of the null protein
        if in_report_only:
            df = df[df.pro_report.astype(bool) == True]
        if not include_decoys:
            df = df[df.pro_is_decoy.astype(bool) == False]

        if len(df) == 0:
            return df

        def pep_ptms(pep):
            # return just the ptms that are located in this pep
            if not pep.pro_ptm_locs:
                return ""
            ptms = pep.pro_ptm_locs.split(";")
            pep_ptms = []
            for p in ptms:
                if pep.pep_start <= (int(p) - 1) < pep.pep_stop:
                    pep_ptms += [p]
            return ";".join(pep_ptms)

        df.pro_ptm_locs = df.apply(pep_ptms, axis=1)
        if ptm_peps_only:
            df = df[df.pro_ptm_locs.astype(bool)]
            # If proteins in this run have no PTMs, the df will now be empty.
            if len(df) == 0:
                return df

        df["n_pep_ptms"] = df.apply(
            lambda x: len(x.pro_ptm_locs.split(";")) if x.pro_ptm_locs else 0, axis=1
        )

        df = df[
            self.peps_columns + ["pro_id", "pro_ptm_locs", "n_pep_ptms"]
        ].reset_index(drop=True)

        if ptms_to_rows:
            new_df = pd.DataFrame(
                df.pro_ptm_locs.str.split(";").tolist(), index=df.pep_i
            ).stack()
            new_df = new_df.reset_index([0, "pep_i"])
            new_df.columns = ["pep_i", "ptm"]
            df = df.set_index("pep_i").join(new_df.set_index("pep_i")).reset_index()

        return df

    def peps__pepseqs(self):
        return (
            self._peps.set_index("pep_i")
            .join(self._pep_seqs.set_index("pep_i"))
            .reset_index()
        )

    def pepseqs__with_decoys(self):
        return self._pep_seqs

    def pepseqs__no_decoys(self):
        return (
            self.pros__no_decoys()
            .set_index("pro_i")
            .join(self.peps__pepseqs().set_index("pro_i"), how="left")
            .reset_index()[self.pep_seqs_columns]
        )

    def pepstrs(self):
        return (
            self._pep_seqs.groupby("pep_i")
            .apply(lambda x: x.aa.str.cat())
            .reset_index()
            .set_index("pep_i")
            .sort_index()
            .rename(columns={0: "seqstr"})
            .reset_index()
        )

    def prostrs(self):
        return (
            self._pro_seqs.groupby("pro_i")
            .apply(lambda x: x.aa.str.cat())
            .reset_index()
            .rename(columns={0: "seqstr"})
        )

    def peps__pepstrs(self):
        return (
            self.peps()
            .set_index("pep_i")
            .join(self.pepseqs().set_index("pep_i"))
            .groupby("pep_i")
            .apply(lambda x: x.aa.str.cat())
            .reset_index()
            .set_index("pep_i")
            .sort_index()
            .rename(columns={0: "seqstr"})
            .reset_index()
        )

    def pros__peps__pepstrs(self):
        return (
            self.pros__peps()
            .set_index("pep_i")
            .join(self.pepstrs().set_index("pep_i"))
            .sort_index()
            .reset_index()
        )

    def __repr__(self):
        try:
            return (
                "PrepResult:\n\nPros:\n"
                + self._pros.to_string()
                + "\n\nPeps:\n"
                + self._peps.to_string()
            )
        except Exception:
            return "PrepResult"

    @classmethod
    def stub_prep_result(cls, pros, pro_is_decoys, peps, pep_pro_iz):
        """
        Make a test stub given a list of pro and pep strings
        """
        _pros = pd.DataFrame(
            [
                (f"id_{i}", is_decoy, i, "", 0)
                for i, (_, is_decoy) in enumerate(zip(pros, pro_is_decoys))
            ],
            columns=PrepResult.pros_columns,
        )

        _pro_seqs = pd.DataFrame(
            [(pro_i, aa) for pro_i, pro in enumerate(pros) for aa in list(pro)],
            columns=PrepResult.pro_seqs_columns,
        )

        _peps = pd.DataFrame(
            [(i, 0, 0, pro_i) for i, (_, pro_i) in enumerate(zip(peps, pep_pro_iz))],
            columns=PrepResult.peps_columns,
        )

        _pep_seqs = pd.DataFrame(
            [(pep_i, aa, 0) for pep_i, pep in enumerate(peps) for aa in list(pep)],
            columns=PrepResult.pep_seqs_columns,
        )

        return PrepResult(
            _pros=_pros, _pro_seqs=_pro_seqs, _peps=_peps, _pep_seqs=_pep_seqs,
        )
