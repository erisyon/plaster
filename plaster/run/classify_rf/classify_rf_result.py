import numpy as np
from plaster.tools.schema import check
import pandas as pd
from plaster.tools.utils import utils
from plaster.run.base_result import BaseResult
from plaster.run.classify_rf.classify_rf_params import ClassifyRFParams


class ClassifyRFResult(BaseResult):
    name = "classify_rf"
    filename = "classify_rf.pkl"

    required_props = dict(
        params=ClassifyRFParams,
        pred_pep_iz=np.ndarray,
        scores=np.ndarray,
        all_class_scores=np.ndarray,
    )

    def __repr__(self):
        try:
            return f"ClassifyRFResult with average score {np.mean(self.scores)} with {len(self.pred_pep_iz)} calls"
        except Exception:
            return "ClassifyRFResult"
