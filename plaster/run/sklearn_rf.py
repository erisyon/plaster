"""
A wrapper for parallelizing SciKitLearn's Random Forest Classifier.

TASK:
    This is to be deprecated once the Nearest-Neighbor classifier
    improves.  The RF is excellent but its memory demands grow
    quadratically with the number of classes and becomes unwieldy
    at proteome scales.
"""
from munch import Munch
import numpy as np
from plaster.tools.utils import utils
from plaster.tools.zap import zap
from plaster.tools.log.log import debug


def _do_predict(classifier, X):
    """
    The Scikit Learn Random Classifier has a predict() and a predict_proba()
    functions. But the expensive part is the predict_proba().
    Oddly, the predict() calls predict_proba() so I was previously doing
    the expensive part twice. Since I only want a single call per row
    I need to re-implement the predict() which
    is nothing more than taking the best score on each row.

    Note, this must be a top-level function so that it can pickle.
    """
    classifier.n_jobs = 1
    all_class_scores = classifier.predict_proba(X)
    argmax_score_per_row = np.argmax(all_class_scores, axis=1)
    maxscore_per_row = all_class_scores[
        np.arange(all_class_scores.shape[0]), argmax_score_per_row
    ]
    class_per_row = classifier.classes_.take(argmax_score_per_row, axis=0)
    return class_per_row, maxscore_per_row, all_class_scores


class SciKitLearnRandomForestClassifier:
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier  # Defer slow import

        self.classifier_args = utils.set_defaults(
            kwargs,
            n_estimators=10,
            min_samples_leaf=50,
            n_jobs=-1,
            max_depth=None,
            max_features="auto",
            max_leaf_nodes=None,
        )

        self.n_progress_jobs = self.classifier_args["n_estimators"]
        self.classifier = RandomForestClassifier(**self.classifier_args)

    def train(self, X, y, progress=None):
        self.classifier.fit(X, y)

    def info(self):
        return Munch(
            feature_importances=self.classifier.feature_importances_,
            n_outputs=self.classifier.n_outputs_,
            n_features=self.classifier.n_features_,
            n_classes=self.classifier.n_classes_,
        )

    def classify(self, test_X, keep_all_class_scores, progress=None):

        # TASK: There's some work to be done here to optimize the size
        #  of this split to dial the memory usage

        n_rows = test_X.shape[0]

        if n_rows < 100:
            pred_y, scores, all_class_scores = _do_predict(
                classifier=self.classifier, X=test_X
            )
        else:
            n_work_orders = n_rows // 100

            with zap.Context(progress=progress, trap_exceptions=False):
                results = zap.work_orders(
                    [
                        Munch(classifier=self.classifier, X=X, fn=_do_predict)
                        for X in np.array_split(test_X, n_work_orders, axis=0)
                    ]
                )
            pred_y = utils.listi(results, 0)
            scores = utils.listi(results, 1)
            all_class_scores = utils.listi(results, 2)
            pred_y = np.concatenate(pred_y)
            scores = np.concatenate(scores)
            if keep_all_class_scores:
                all_class_scores = np.concatenate(all_class_scores)

        if not keep_all_class_scores:
            all_class_scores = None

        return pred_y, scores, all_class_scores
