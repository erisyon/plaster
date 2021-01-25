import numpy as np
import pandas as pd
from plaster.run.base_result import ArrayResult, BaseResult
from plaster.run.nn_v2.nn_v2_params import NNV2Params
from plaster.tools.log.log import debug


class NNV2Result(BaseResult):
    name = "nn_v2"
    filename = "nn_v2.pkl"

    columns = (
        "peak_i",
        "dyt_i",
        "pep_i",
        "dyt_score",
        "score",
        "k",
        "logp_dyt",
        "logp_k",
    )

    # * "calls" are the "best" assignment for each radrow based on top score.
    # * "all" is a structrure (potentially much larger) of each radrow compare
    #   to every dyetrack (dyt).

    required_props = dict(
        params=NNV2Params,
        _test_calls=(type(None), pd.DataFrame),
        _train_calls=(type(None), pd.DataFrame),
        _sigproc_calls=(type(None), pd.DataFrame),
        # "all_calls" are created when params.against_all_dyetracks_output is True
        _test_all=(type(None), pd.DataFrame),
        _train_all=(type(None), pd.DataFrame),
        _sigproc_all=(type(None), pd.DataFrame),
        _test_peps_pr=(type(None), pd.DataFrame),
        _test_peps_pr_abund=(type(None), pd.DataFrame),
        _dyemat=(type(None), np.ndarray),
        _dyepeps=(type(None), np.ndarray),
    )

    @property
    def test_peps_pr(self):
        return self._test_peps_pr

    @property
    def test_test_peps_pr_abund(self):
        return self._test_peps_pr_abund

    @property
    def includes_test_results(self):
        return self._test_calls is not None

    @property
    def includes_train_results(self):
        return self._train_calls is not None

    @property
    def includes_sigproc_results(self):
        return self._sigproc_calls is not None

    @property
    def includes_all_results(self):
        return self._test_all is not None

    def __repr__(self):
        try:
            return (
                f"NNV2Result "
                f"({'includes' if self.includes_test_results else 'does not include'} test results) "
                f"({'includes' if self.includes_train_results else 'does not include'} train results) "
                f"({'includes' if self.includes_sigproc_results else 'does not include'} sigproc results)"
            )
        except:
            return "NNV2Result"

    @classmethod
    def filter(cls, df, include_nul_calls=False, k_range=None, score_range=None):
        mask = np.ones((len(df),), dtype=bool)

        if not include_nul_calls:
            mask = (df.dyt_i > 0) & (df.pep_i > 0)

        if k_range is not None:
            if k_range[0] is not None:
                mask &= k_range[0] <= df.k
            if k_range[1] is not None:
                mask &= df.k <= k_range[1]

        if score_range is not None:
            if score_range[0] is not None:
                mask &= score_range[0] <= df.score
            if score_range[1] is not None:
                mask &= df.score <= score_range[1]

        return mask

    def calls(self, dataset="test", **kwargs):
        """
        Returns a "call" dataframe with optional filter conditions.
        Example:
            df = nn_v2_result.df(k_range=(0.5, 1.5))
        """
        df = getattr(self, f"_{dataset}_calls")
        mask = self.filter(df, **kwargs)
        return df[mask]

    def all(self, dataset="test", **kwargs):
        """
        Like "calls" but for the "all" set. Only available when the classifier
        was run with the "against_all_dyetracks_output" True.
        """
        df = getattr(self, f"_{dataset}_all")
        mask = self.filter(df, **kwargs)
        return df[mask]
