import numpy as np
import pandas as pd
from plaster.run.base_result import ArrayResult, BaseResult
from plaster.run.nn_v2.nn_v2_params import NNV2Params


class NNV2Result(BaseResult):
    name = "nn_v2"
    filename = "nn_v2.pkl"

    required_props = dict(
        params=NNV2Params,
        test_pred_pep_iz=np.ndarray,
        test_pred_dye_iz=np.ndarray,
        test_scores=np.ndarray,
        test_true_pep_iz=(type(None), np.ndarray),
        test_peps_pr=(type(None), pd.DataFrame),
        test_peps_pr_abund=(type(None), pd.DataFrame),
        sigproc_pred_pep_iz=(type(None), np.ndarray),
        sigproc_scores=(type(None), np.ndarray),
        sigproc_pred_dye_i=(type(None), np.ndarray),
        # Things in nn_v1_result
        # test_dt_mat=np.ndarray,
        # test_dyetracks_df=pd.DataFrame,
        # test_dt_pep_sources_df=pd.DataFrame,
        # test_true_dt_iz=np.ndarray,
        # test_pred_dt_iz=np.ndarray,
        # test_dt_scores=np.ndarray,
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
