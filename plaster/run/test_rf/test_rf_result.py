import numpy as np
import pandas as pd
from plaster.run.base_result import BaseResult
from plaster.run.test_rf.test_rf_params import TestRFParams


class TestRFResult(BaseResult):
    name = "test_rf"
    filename = "test_rf.pkl"

    required_props = dict(
        params=TestRFParams,
        test_true_pep_iz=np.ndarray,
        test_pred_pep_iz=np.ndarray,
        test_scores=np.ndarray,
        test_all_class_scores=(type(None), np.ndarray),
        test_peps_pr=(type(None), pd.DataFrame),
        test_peps_pr_abund=(type(None), pd.DataFrame),
        train_true_pep_iz=(type(None), np.ndarray),
        train_pred_pep_iz=(type(None), np.ndarray),
        train_scores=(type(None), np.ndarray),
        train_all_class_scores=(type(None), np.ndarray),
        train_peps_pr=(type(None), pd.DataFrame),
        train_peps_pr_abund=(type(None), pd.DataFrame),
    )

    def includes_train_results(self):
        return self.train_pred_pep_iz is not None

    def __repr__(self):
        try:
            return (
                f"TestRFResult with average score {np.mean(self.test_scores)} "
                f"({'includes' if self.includes_train_results else 'does not include'} train results)"
            )
        except:
            return "TestRFResult"
