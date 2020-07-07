from plaster.run.base_result import BaseResult
from plaster.run.train_rf.train_rf_params import TrainRFParams
from plaster.run.sklearn_rf import SciKitLearnRandomForestClassifier


class TrainRFResult(BaseResult):
    name = "train_rf"
    filename = "train_rf.pkl"

    required_props = dict(
        params=TrainRFParams, classifier=SciKitLearnRandomForestClassifier,
    )

    def __repr__(self):
        return f"TrainRFResult"
