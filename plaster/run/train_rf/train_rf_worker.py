"""
Train the Random Forest classifier.

TASK:
    This is to be deprecated once the Nearest-Neighbor classifier
    improves.  The RF is excellent but its memory demands grow
    quadratically with the number of classes and becomes unwieldy
    at proteome scales.

"""

import numpy as np
from plaster.run.sklearn_rf import SciKitLearnRandomForestClassifier
from plaster.run.train_rf.train_rf_result import TrainRFResult
from plaster.tools.log.log import important
from plaster.tools.utils import data


def _subsample(n_subsample, X, y):
    n_peps = np.max(y) + 1
    _y = []
    _X = []
    for pep_i in range(n_peps):
        args = np.argwhere(y == pep_i)
        arg_subsample = data.subsample(args, n_subsample)
        _X += [X[arg_subsample]]
        _y += [y[arg_subsample]]

    X = np.vstack(_X)
    y = np.vstack(_y)
    return np.squeeze(X, axis=1), np.squeeze(y, axis=1)


def train_rf(train_rf_params, sim_result, progress=None):
    X = sim_result.flat_train_radmat()
    y = sim_result.train_true_pep_iz()

    if train_rf_params.n_subsample is not None:
        X, y = _subsample(train_rf_params.n_subsample, X, y)

    else:
        if sim_result.params.n_samples_train > 1000:
            important(
                "Warning: RF does not memory-scale well when the n_samples_train is > 1000."
            )

    del train_rf_params["n_subsample"]
    classifier = SciKitLearnRandomForestClassifier(**train_rf_params)
    classifier.train(X, y, progress)
    return TrainRFResult(params=train_rf_params, classifier=classifier)
