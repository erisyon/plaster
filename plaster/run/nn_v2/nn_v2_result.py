import numpy as np
import pandas as pd
from plaster.run.base_result import ArrayResult, BaseResult
from plaster.run.nn_v2.nn_v2_params import NNV2Params


class NNV2Result(BaseResult):
    name = "nn_v2"
    filename = "nn_v2.pkl"

    required_props = dict(
        params=NNV2Params,
        # test_pred_dt_iz=ArrayResult,
        # test_dt_scores=ArrayResult,
        test_pred_pep_iz=np.ndarray,
        test_scores=np.ndarray,
        # test_peps_pr=(type(None), pd.DataFrame),
        # test_peps_pr_abund=(type(None), pd.DataFrame),
        # Optional: when the testing is also run on training to evaluate overfitting...
        # train_pred_dt_iz=(type(None), ArrayResult),
        # train_dt_scores=(type(None), ArrayResult),
        # train_pred_pep_iz=(type(None), ArrayResult),
        # train_scores=(type(None), ArrayResult),
        # train_peps_pr=(type(None), pd.DataFrame),
        # train_peps_pr_abund=(type(None), pd.DataFrame),
    )

    def includes_train_results(self):
        return self.train_pred_dt_iz is not None

    def __repr__(self):
        try:
            return (
                f"NNV2Result with average score {np.mean(self.test_scores)} "
                f"({'includes' if self.includes_train_results else 'does not include'} train results)"
            )
        except:
            return "NNV2Result"
