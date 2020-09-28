import traceback

import numpy as np
import pandas as pd
from munch import Munch
from plaster.tools.aaseq.aaseq import aa_str_to_list
from plaster.tools.log.log import debug, exception, info, prof
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.utils.data import ConfMat
from plaster.tools.zap import zap


def _do_pep_pr_curve(bag, pep_i):
    return (int(pep_i), bag.pr_curve_pep(pep_iz_subset=[pep_i]))


def _do_false_rates_by_pep(pep_i, bag, at_prec, n_false):
    return bag.false_rates_by_pep(pep_i, at_prec, n_false)


def _do_peps_above_thresholds(x, precision, recall):
    return np.any((x.prec > precision) & ((x.recall > recall)))


class CallBag:
    @staticmethod
    def _prs_at_prec(at_p, p, r, s):
        """
        Helper to get the prec, recall, and score at or above a specific precision.
        In the list of (p, r, s) find the first location from the left (high scores first
        in list) in p where p > at_p and return the p, r, s there.

        Because the functions tends to be noisy on the left (large scores), search from
        the right looking for the first place that p exceeds at_p then gobble
        up any ties (same s).

        p = [0.3, 0.2, 0.2, 0.2, 0.1]
        r = [0.1, 0.2, 0.3, 0.4, 0.5]
        s = [0.9, 0.8, 0.7, 0.6, 0.5]
        _prs_at_prec(0.2, p, r, s) == (0.2, 0.4, 0.6)

        Note: Numpy argmax is a tricky beast for booleans. The docs say:
            In case of multiple occurrences of the maximum values,
            the indices corresponding to the first occurrence are returned.

        So it can occur that ALL the bools in the mask are True or False
        and then it can be confusing.

        Example:

            v = ?
            p = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
            s = (p >= v).sum()
            a = np.argmax(p >= v)

            Suppose v == 0.0; The whole array (p >= v) is True and argmax is 0.
            That makes sense.

            Now consider v == 1.0:
            Now NO value of p is >= 1.0 so the whole array (p >= v) is False
            and thus the rule of the doc says: "indices corresponding to the first
            occurrence are returned". So return return value is 0 still!!!

        Thus, when using argmax as a search you must consider the sum also.

        returns:
            (prec, recall, score) here prec >= at_p
        """

        if len(p) == 0:
            # Can happen when the bag is empty
            return 0.0, 0.0, 0.0

        assert np.all(np.diff(s) <= 0.0)

        # REVERSE to search from the lowest values (less noise)
        rev_p = p[::-1]
        rev_s = s[::-1]
        rev_mask = rev_p >= at_p

        if rev_mask.sum() == 0.0:
            # No value of the p array is grater than the at_p target value.
            return 0.0, 0.0, 0.0

        right_most_pos = np.argmax(rev_mask)
        score_at_pos = rev_s[right_most_pos]

        # BREAK ties by finding the left-most value with the same score
        # Note that this is in the ORIGINAL order!
        left_most_pos = np.argmax(s == score_at_pos)

        return p[left_most_pos], r[left_most_pos], s[left_most_pos]

    def __init__(
        self,
        prep_result=None,
        sim_result=None,
        all_class_scores=None,
        cached_pr=None,
        cached_pr_abund=None,
        classifier_name=None,
        **kwargs,
    ):
        self._prep_result = prep_result
        self._sim_result = sim_result
        self._all_class_scores = all_class_scores
        self._cached_pr = cached_pr
        self._cached_pr_abund = cached_pr_abund
        self.classifier_name = classifier_name
        if sim_result is not None:
            assert len(self._sim_result.train_pep_recalls) == self._prep_result.n_peps
        self.df = pd.DataFrame(kwargs)

    def copy(self):
        """
        return a new CallBag copied from this one where the row mask is True
        Note the cached PR information is not copied.
        """
        return CallBag(
            prep_result=self._prep_result,
            sim_result=self._sim_result,
            true_pep_iz=np.copy(self.df.true_pep_iz.values),
            pred_pep_iz=np.copy(self.df.pred_pep_iz.values),
            scores=np.copy(self.df.scores.values),
        )

    def filter_by_mask(self, mask):
        """
        return a new CallBag copied from this one where the row mask is True
        """
        return CallBag(
            prep_result=self._prep_result,
            sim_result=self._sim_result,
            true_pep_iz=self.df.true_pep_iz[mask],
            pred_pep_iz=self.df.pred_pep_iz[mask],
            scores=self.df.scores[mask],
        )

    def sample(self, n_samples):
        """return a new CallBag by randomly sampling n_sample rows"""
        mask = np.zeros((self.n_rows,))
        mask[np.random.choice(self.n_rows, n_samples)] = 1
        return self.filter_by_mask(mask > 0)

    def correct_call_iz(self):
        """
        return an array of indices where the call were correct
        """
        return np.argwhere(
            self.df.true_pep_iz.values == self.df.pred_pep_iz.values
        ).flatten()

    def incorrect_call_iz(self):
        """
        return an array of indices where the call were incorrect
        """
        return np.argwhere(
            self.df.true_pep_iz.values != self.df.pred_pep_iz.values
        ).flatten()

    def correct_call_mask(self):
        return self.df.true_pep_iz.values == self.df.pred_pep_iz.values

    def incorrect_call_mask(self):
        return self.df.true_pep_iz.values != self.df.pred_pep_iz.values

    @property
    def n_rows(self):
        return len(self.df)

    @property
    def true_pep_iz(self):
        return self.df.true_pep_iz.values

    @property
    def n_peps(self):
        return self._prep_result.n_peps

    @property
    def pred_pep_iz(self):
        return self.df.pred_pep_iz.values

    @property
    def scores(self):
        return self.df.scores.values

    def average_classifier_scores_for_class(
        self, klass_i, for_true_klass=True, max_likelihood=False
    ):
        """
        If all_class_scores is available:
        Returns a vector of individually averaged proba scores per class assigned by
        the classifer for the true or predicted klass_i.

        If all_class_scores is not available, or max_likelihood has been specified:
        Returns average of the max score for klass_i, true or pred.

        returns: number of calls, and a vector of average score(s)
        """

        column = "true_pep_iz" if for_true_klass else "pred_pep_iz"
        idx = self.df.index[self.df[column] == klass_i]
        n_calls = len(idx)

        if max_likelihood or self._all_class_scores is None:
            avg_scores = [self.df.scores[idx].mean()]
        else:
            scores = self._all_class_scores[idx]
            avg_scores = np.sum(scores, axis=0) / scores.shape[0]

        return n_calls, avg_scores

    def true_peps__pros(self):
        self.df["order"] = np.arange(len(self.df))
        return (
            self.df[["order", "true_pep_iz"]]
            .rename(columns=dict(true_pep_iz="pep_i"))
            .set_index("pep_i")
            .join(self._prep_result.pros__peps().set_index("pep_i"), how="left")
            .sort_values("order")
            .reset_index()
        )

    def true_peps__unique(self):
        return (
            self.df[["true_pep_iz"]]
            .rename(columns=dict(true_pep_iz="pep_i"))
            .set_index("pep_i")
            .join(self._prep_result.pepstrs().set_index("pep_i"), how="left")
            .drop_duplicates()
            .reset_index()
        )

    def pred_peps__pros(self):
        self.df["order"] = np.arange(len(self.df))
        return (
            self.df[["order", "pred_pep_iz"]]
            .rename(columns=dict(pred_pep_iz="pep_i"))
            .set_index("pep_i")
            .join(self._prep_result.pros__peps().set_index("pep_i"), how="left")
            .sort_values("order")
            .reset_index()
        )

    def pred_peps__unique(self):
        return (
            self.df[["pred_pep_iz"]]
            .rename(columns=dict(pred_pep_iz="pep_i"))
            .set_index("pep_i")
            .join(self._prep_result.pepstrs().set_index("pep_i"), how="left")
            .drop_duplicates()
            .reset_index()
        )

    def conf_mat(
        self, true_set_size=None, pred_set_size=None, mask=None,
    ):
        """
        Build a confusion matrix from the call bag.

        If the set_size parameters are not given it
        will determine those sizes by asking the prep_result.
        """
        true = self.df["true_pep_iz"].values
        pred = self.df["pred_pep_iz"].values

        # Compute true_set_size and pred_set_size if they are not specified
        if true_set_size is None:
            true_set_size = self._prep_result.n_peps

        if pred_set_size is None:
            pred_set_size = self._prep_result.n_peps

        n_rows = len(self.df)
        if mask is not None:
            if isinstance(mask, pd.Series):
                mask = mask.values
            check.array_t(mask, shape=(n_rows,), dtype=np.bool_)
            pred = np.copy(pred)
            pred[~mask] = 0

        return ConfMat.from_true_pred(true, pred, true_set_size, pred_set_size)

    @staticmethod
    def _auc(x, y):
        """A simple rectangular (Euler) integrator. Simpler and easier than sklearn metrics"""
        zero_padded_dx = np.concatenate(([0], x))
        return (np.diff(zero_padded_dx) * y).sum()

    def pr_curve_pep(self, pep_iz_subset=None, n_steps=50):
        """
        See: https://docs.google.com/document/d/1MW92KNTaNtuL1bR_p0U1FwfjaiomHD3fRldiSVF74pY/edit#bookmark=id.4nqatzscuyw7

        Unlike sklearn's implementation, this one samples scores
        uniformly to prevent returning gigantic arrays.

        Returns a tuple of arrays; each row of the arrays is an increasing score threshold. The arrays are:
            * precision, recall, score_thresh, area_under_curve
        """

        # Obtain a reverse sorted calls: true, pred, score
        true = self.df["true_pep_iz"].values
        pred = self.df["pred_pep_iz"].values
        scores = self.df["scores"].values

        # At this point true, pred, scores are sorted WHOLE SET OF ALL PEPTIDES

        # If a subset is not requested then assume ALL are wanted
        if pep_iz_subset is None:
            pep_iz_subset = np.unique(
                np.concatenate((self.df.true_pep_iz[1:], self.df.pred_pep_iz))
                # 1: => don't include the null peptide class from true
            )

        # MASK calls in the subset
        true_in_subset_mask = np.isin(true, pep_iz_subset)
        pred_in_subset_mask = np.isin(pred, pep_iz_subset)

        return self.pr_curve_either(
            "pep",
            true,
            pred,
            scores,
            true_in_subset_mask,
            pred_in_subset_mask,
            level_iz_subset=pep_iz_subset,
            n_steps=n_steps,
        )

    def pr_curve_pro(self, pro_iz_subset=None, n_steps=50):
        """
        Similar format to pr_curve_pep, but is PR curve for proteins, not peptides
        Returns a tuple of arrays; each row of the arrays is an increasing score threshold. The arrays are:
            * precision, recall, score_thresh, area_under_curve
        """

        # Obtain a reverse sorted calls: true, pred, score
        true = self.true_peps__pros().pro_i.values
        pred = self.pred_peps__pros().pro_i.values
        scores = self.df["scores"].values

        # At this point true, pred, scores are sorted WHOLE SET OF ALL PROTEINS

        # If a subset is not requested then assume ALL are wanted
        if pro_iz_subset is None:
            if len(true) > 0 and len(pred) > 0:
                pro_iz_subset = np.unique(
                    np.concatenate((np.array(true), np.array(pred)))
                )
            elif len(pred) > 0:
                pro_iz_subset = np.unique(np.array(true))
            else:
                pro_iz_subset = np.array([])

        # MASK calls in the subset
        true_in_subset_mask = np.isin(true, pro_iz_subset)
        pred_in_subset_mask = np.isin(pred, pro_iz_subset)

        return self.pr_curve_either(
            "pro",
            true,
            pred,
            scores,
            true_in_subset_mask,
            pred_in_subset_mask,
            level_iz_subset=pro_iz_subset,
            n_steps=n_steps,
        )

    def pr_curve_either(
        self,
        level,
        true,
        pred,
        scores,
        true_in_subset_mask,
        pred_in_subset_mask,
        level_iz_subset,
        n_steps=50,
    ):
        # Throughout this function, "level" (e.g. level_iz_subset) refers to "peptide or protein" (or
        # potentially, in the future, PTM or whatever else we might make a PR curve of)

        # At this point, true_ and pred_in_subset_mask are masks on the original set.
        # We now reduce to the set of interest so that we sort a smaller set

        true_or_pred_subset_mask = true_in_subset_mask | pred_in_subset_mask
        true = true[true_or_pred_subset_mask]
        pred = pred[true_or_pred_subset_mask]
        scores = scores[true_or_pred_subset_mask]
        true_in_subset_mask = true_in_subset_mask[true_or_pred_subset_mask]
        pred_in_subset_mask = pred_in_subset_mask[true_or_pred_subset_mask]

        # Now sort on a smaller set
        sorted_iz = np.argsort(scores)[::-1]
        true = true[sorted_iz]
        pred = pred[sorted_iz]
        scores = scores[sorted_iz]

        true_in_subset_mask = true_in_subset_mask[sorted_iz]
        pred_in_subset_mask = pred_in_subset_mask[sorted_iz]

        # How many true are in the subset? This will be
        # used as the denominator of recall.
        n_true_in_subset = true_in_subset_mask.sum()

        # WALK through scores linearlly from high to low, starting
        # at (1.0 - step_size) so that the first group has contents.
        step_size = 1.0 / n_steps

        # prsa sdtands for "Precision Recall Score Area_under_curve"
        prsa = np.zeros((n_steps, 4))
        precision_column = 0
        recall_column = 1
        score_thresh_column = 2
        auc_column = 3

        for prsa_i, score_thresh in enumerate(np.linspace(1 - step_size, 0, n_steps)):
            # i is the index where *ALL* scores before this point are greater
            # than or equal to the score_thresh. Note that because many calls
            # may have *tied* scores, we use np_arg_last_where to pick the
            # *last* position (ie lowest score) where the statement is true.
            i = utils.np_arg_last_where(scores >= score_thresh)
            if i is None:
                prsa[prsa_i] = (0.0, 0.0, score_thresh, 0.0)
            else:
                correct_at_i_mask = true[0 : i + 1] == pred[0 : i + 1]
                pred_at_i_mask = pred_in_subset_mask[0 : i + 1]

                # At i, count:
                #  * How many of the subset of interest have been predicted?
                #    This will be used as the denominator of precision.
                #  * How many correct calls of the subset have been made?
                #    This is the numerator of precision and recall.
                #  Note that for the correct, the masking doesn't matter if
                #  we choose the true_mask or pred_mask because they are they same
                #  in the case of a correct call.
                n_pred_at_i = pred_at_i_mask.sum()
                n_correct_and_in_subset_at_i = (
                    correct_at_i_mask  # & pred_at_i_mask
                ).sum()

                prsa[prsa_i] = (
                    # Precision: Fraction of those that were called apples at i that were in fact apples
                    utils.np_safe_divide(n_correct_and_in_subset_at_i, n_pred_at_i),
                    # Recall: Fraction of all apples that were called apples at i
                    utils.np_safe_divide(
                        n_correct_and_in_subset_at_i, n_true_in_subset
                    ),
                    # Score threshold is stepping down linearly
                    score_thresh,
                    0.0,
                )
                # The Area under the curve up to this point (requires two points)
                prsa[prsa_i, auc_column] = self._auc(
                    prsa[0 : prsa_i + 1, recall_column],
                    prsa[0 : prsa_i + 1, precision_column],
                )

        # CORRECT for the prior-recall.
        # During simulation some rows may be all-dark.
        # Those are accounted for here by scaling down the recall by
        # the fraction of non-dark rows / all rows.
        # This is done as MEAN of all recalls over the set of interest.

        # EXTRACT training recalls from the subset of peps or pros.
        # This will leave NANs for all those that are not in the subset.
        if self._sim_result is not None and level == "pep":
            filtered_level_recalls = np.full_like(
                self._sim_result.train_pep_recalls, np.nan
            )
            filtered_level_recalls[
                level_iz_subset
            ] = self._sim_result.train_pep_recalls[level_iz_subset]
        elif self._sim_result is not None and level == "pro":
            pro_pep_df = self._prep_result.pros__peps()
            n_pro = pro_pep_df.pro_id.nunique()
            filtered_level_recalls = np.full((n_pro,), np.nan)
            filtered_level_recalls[level_iz_subset] = 0.0
            for protein_index in level_iz_subset:
                for peptide_index in pro_pep_df.loc[
                    pro_pep_df["pro_i"] == protein_index
                ].pep_i:
                    # NOTE: we are assuming here that the recall of the best performing peptide of a given
                    #      protein is a good approximation of the protein as a whole.  Some initial literature
                    #      research indicates this is a decent starting approximation, but it may need to
                    #      be revisited in the future.
                    filtered_level_recalls[protein_index] = max(
                        filtered_level_recalls[protein_index],
                        self._sim_result.train_pep_recalls[peptide_index],
                    )
        else:
            filtered_level_recalls = np.full((prsa.shape[0],), 1.0)

        # Use nanmean to ignore al those nans (the peps not in the subset)
        # And then use np.nan_to_num in case the subset was empty, we want 0 not nan
        mean_recall = np.nan_to_num(np.nanmean(filtered_level_recalls))
        assert 0.0 <= mean_recall <= 1.0

        # SCALE-DOWN all recall
        prsa[:, recall_column] *= mean_recall

        # SKIP all initial rows where the recall is zero, these clutter up the graph
        # The return may thus have fewer than n_steps rows.
        first_non_zero_i = utils.np_arg_first_where(prsa[:, recall_column] > 0.0)

        filtered_prsa = prsa[first_non_zero_i:]

        assert np.all(np.diff(filtered_prsa[:, 2]) <= 0.0)

        return (
            filtered_prsa[:, 0],  # Precision
            filtered_prsa[:, 1],  # Recall
            filtered_prsa[:, 2],  # Score thresholds
            filtered_prsa[:, 3],  # AUC
        )

    def pr_curve_by_pep(
        self, return_auc=False, pep_iz=None, force_compute=False, progress=None
    ):
        """
        Obtain pr_curves for every peptide.

        If all params are default, may returned cached information computed
        during the run.

        Returns:
            A (potentially HUGE) df of every P/R for every peptide
            A smaller df with just the pep_i and the Area-Under-Curve

        This uses the work_order system (as opposed to the
        higher-level array_split_map()) because the _do_pep_pr_curve
        returns 3 identical returns AND one scalar; array_split_map() doesn't
        like that.
        """

        # The PR for all peptides is computed during the run (no auc).
        if not return_auc and not force_compute and self._cached_pr is not None:
            df = self._cached_pr
            if pep_iz is not None:
                df = df[df.pep_i.isin(pep_iz)]
            return df.copy()

        if pep_iz is None:
            pep_iz = self._prep_result.peps().pep_i.values
        if isinstance(pep_iz, np.ndarray):
            pep_iz = pep_iz.tolist()
        check.list_t(pep_iz, int)

        # prof()
        results = zap.work_orders(
            [Munch(fn=_do_pep_pr_curve, pep_i=pep_i, bag=self,) for pep_i in pep_iz],
            _process_mode=False,
            _trap_exceptions=False,
            # _progress=progress,
        )
        # prof("prs")

        df_per_pep = [
            pd.DataFrame(
                dict(
                    pep_i=np.repeat(np.array([pep_i]), prec.shape[0]),
                    prec=prec,
                    recall=recall,
                    score=score,
                )
            )
            for pep_i, (prec, recall, score, _) in results
        ]

        if len(df_per_pep) > 0:
            pr_df = pd.concat(df_per_pep, axis=0)
        else:
            pr_df = None

        auc_df = pd.DataFrame(
            [(pep_i, auc) for pep_i, (_, _, _, auc) in results],
            columns=["pep_i", "auc"],
        )

        if return_auc:
            return pr_df, auc_df
        else:
            return pr_df

    def pr_curve_by_pep_with_abundance(
        self,
        return_auc=False,
        pep_iz=None,
        n_steps=50,
        pep_abundance=None,
        force_compute=False,
        progress=None,
    ):
        """
        In principle the same computation as pr_curve_by_pep (which uses pr_curve_pep())
        but here is done via a confusion matrix which makes it possible to factor
        in peptide abundance information.  This also means that this function is
        inherently parallel in that PR is computed for all pep_iz at once via the
        conf_mat routines precision() and recall() -- so not sure if it is worth
        trying to parallelize further (on n_steps?)

        DHW 9/28/2020 - I profiled some of the calls in the n_steps loops, and in general, on my machine,
                        self.conf_mat_at_score_threshold and conf_mat.scale_by_abundance are each on the order of 1 second,
                        and conf_mat.precision()[pep_iz], conf_mat.recall()[pep_iz], and calculating auc are each on the order of 100m
        """

        # TODO: write some tests that assert these two fns return the same
        #       values for individual peptide PR

        # PR with abundance is calculated during a run if abundance was avail, and
        # cached in CallBag.  If the values passed to us are default, return the
        # cached copy.
        if (
            not force_compute
            and not return_auc
            and pep_abundance is None
            and self._cached_pr_abund is not None
        ):
            df = self._cached_pr_abund
            if pep_iz is not None:
                df = df[df.pep_i.isin(pep_iz)]
            return df.copy()

        # If pep_abundance is None, take the information from PrepResult.
        # If none is available, return None.
        if pep_abundance is None:
            pep_abundance = self._prep_result.peps_abundance()
            if pep_abundance is None:
                return None

        if pep_iz is None:
            pep_iz = self._prep_result.peps().pep_i.values
        if isinstance(pep_iz, np.ndarray):
            pep_iz = pep_iz.tolist()
        check.list_t(pep_iz, int)
        n_peps = len(pep_iz)

        step_size = 1.0 / n_steps

        # prsa stands for "Precision Recall Score Area_under_curve"
        prsa = np.zeros((n_steps, n_peps, 4))
        precision_column = 0
        recall_column = 1
        # score_thresh_column = 2
        # auc_column = 3

        # This loop can take 2-3 minutes
        # Not necessarily advantageous to parallize this as we're probably already parallelized at the run level
        for prsa_i, score_thresh in enumerate(np.linspace(1 - step_size, 0, n_steps)):
            if progress:
                progress(prsa_i, n_steps, retry=False)
            # TODO: could opimize this by subselecting pep_iz for conf_mat if we're not
            # doing all peps - creates smaller confusion matrix.
            conf_mat = self.conf_mat_at_score_threshold(score_thresh)
            assert pep_abundance is not None

            conf_mat = conf_mat.scale_by_abundance(pep_abundance)
            p = conf_mat.precision()[pep_iz]
            r = conf_mat.recall()[pep_iz]
            auc = np.array(
                [
                    self._auc(
                        prsa[0 : prsa_i + 1, p_i, recall_column],
                        prsa[0 : prsa_i + 1, p_i, precision_column],
                    )
                    for p_i in range(n_peps)
                ]
            )
            prsa[prsa_i] = np.transpose([p, r, [score_thresh] * n_peps, auc])

        prsa = np.swapaxes(prsa, 0, 1)

        train_pep_recalls = self._sim_result.train_pep_recalls

        pep_prsa_tuples = []
        for pep_i, pep_prsa in zip(pep_iz, prsa):
            first_non_zero_i = utils.np_arg_first_where(
                pep_prsa[:, recall_column] > 0.0
            )
            pep_prsa_tuples += [
                (
                    pep_prsa[first_non_zero_i:, 0],  # Precision
                    pep_prsa[first_non_zero_i:, 1] * train_pep_recalls[pep_i],  # Recall
                    pep_prsa[first_non_zero_i:, 2],  # Score
                    pep_prsa[first_non_zero_i:, 3],  # AUC
                )
            ]

        # At this point we have a single tuple per peptide, but each entry in
        # the tuple is a list of values.  This is like pr_curve_pep() and potentially
        # nice way to return this information. But let's return a DataFrame so
        # that the output is the same as pr_curve_by_pep()

        prs_df = pd.DataFrame(
            [
                (pep_i, p, r, s)
                for pep_i, prsa in zip(pep_iz, pep_prsa_tuples)
                for p, r, s in zip(prsa[0], prsa[1], prsa[2])
            ],
            columns=["pep_i", "prec", "recall", "score"],
        )

        if return_auc:
            a_df = pd.DataFrame(
                [
                    (pep_i, a)
                    for pep_i, prsa in zip(pep_iz, pep_prsa_tuples)
                    for a in prsa[3]
                ],
                columns=["pep_i", "auc"],
            )
            return prs_df, a_df

        return prs_df

    def score_thresh_for_pep_at_precision(self, pep_i, at_prec, n_steps=200):
        p, r, s, _ = self.pr_curve_pep(pep_iz_subset=[pep_i], n_steps=n_steps)
        """
        Note: returns 0.0 if there's nothing with that precision.
        """
        p, r, s, _ = self.pr_curve_pep(pep_iz_subset=[pep_i])
        assert np.all(np.diff(s) <= 0.0)
        _, _, s_at_prec = CallBag._prs_at_prec(at_prec, p, r, s)
        return s_at_prec

    def conf_mat_at_score_threshold(self, score_thresh):
        return self.conf_mat(mask=self.scores >= score_thresh.astype(self.scores.dtype))

    def false_rates_by_pep(self, pep_i, at_prec, n_false):
        """
        For the given pep, find the top most falses (false-negatives and
        false-positives). This allows us to see which other peptides are
        the worse offenders for collisions with this peptide.

        Arguments:
            pep_i: The peptide to examine
            at_prec: The precision at which this computed
            n_false: The number of false-positives and false-negatives to return

        Returns:
            A DataFrame of:
                pep_i
                at_prec (copied from input)
                recall_at_prec
                score_at_prec
                false_type
                false_pep_i
                false_weight

        Notes:
            To do this we get a score at the specified precision. Because the
            precision can be noisy at high-scores, this is computed by walking
            from the least-to-the most score and choose the first place
            that the precision is >= prec.
        """

        pr_df = self.pr_curve_by_pep(pep_iz=[pep_i], force_compute=False)
        p = pr_df.prec.values
        r = pr_df.recall.values
        s = pr_df.score.values

        assert np.all(np.diff(s) <= 0.0)

        p_at_prec, r_at_prec, s_at_prec = CallBag._prs_at_prec(at_prec, p, r, s)

        # Convert the desired precision into a score so we can make a ConfMat
        # and then use that ConfMat to find the top false calls
        cm = self.conf_mat(mask=self.scores >= s_at_prec)

        false_tuples = cm.false_calls(pep_i, n_false=n_false)
        false_df = pd.DataFrame(
            false_tuples, columns=["false_type", "false_pep_i", "false_weight"]
        )
        false_df = false_df[false_df.false_weight > 0.0]

        false_df["false_i"] = range(len(false_df))
        false_df["pep_i"] = pep_i
        false_df["at_prec"] = at_prec
        false_df["recall_at_prec"] = r_at_prec
        false_df["score_at_prec"] = s_at_prec

        return false_df

    def false_rates_all_peps(self, at_prec, n_false=4):
        pep_iz = self._prep_result.peps().pep_i.values

        return pd.concat(
            zap.arrays(
                _do_false_rates_by_pep,
                dict(pep_i=pep_iz),
                bag=self,
                at_prec=at_prec,
                n_false=n_false,
            )
        ).reset_index(drop=True)

    def false_rates_all_peps__flus(
        self, at_prec, n_false=4, protein_of_interest_only=True
    ):
        flus = self._sim_result.flus()
        pepstrs = self._prep_result.pepstrs()
        pros = self._prep_result.pros()
        peps = self._prep_result.peps()

        df = peps.set_index("pep_i")

        df = df.join(pepstrs.set_index("pep_i")).reset_index()

        df = df.join(flus.set_index("pep_i")).reset_index()

        df = pd.merge(df, pros[["pro_i", "pro_id"]], on="pro_i", how="left")

        def pros_for_flu(row):
            # find each instance flu in all pros
            return ",".join(map(str, df[df.flustr == row.flustr].pro_i))

        df["flu_pros"] = df.apply(pros_for_flu, axis=1)

        df = df.join(
            self.false_rates_all_peps(at_prec, n_false).set_index("pep_i")
        ).reset_index(drop=True)

        # ensure that false_i can be used as a filter to show only one instance per peptide.
        df.false_i = df.false_i.fillna(0).astype(int)

        df[
            "at_prec"
        ] = at_prec  # ensures that at_prec appears even if false_rates_all_peps returns empty df

        df = df.set_index("false_pep_i", drop=False).join(
            peps[["pep_i", "pro_i"]]
            .rename(columns=dict(pro_i="false_pro_i"))
            .set_index("pep_i"),
            how="left",
        )

        keep_flu_cols = ["flustr", "pep_i"]

        df = (
            df.set_index("false_pep_i", drop=False)
            .join(
                flus[keep_flu_cols]
                .rename(columns=dict(flustr="false_flustr"))
                .set_index("pep_i"),
                how="left",
            )
            .reset_index(drop=True)
        )

        if protein_of_interest_only:
            # TODO: this caused a problem when done up top, probably because
            # of no reset_index() or something I don't understand.  Ask Zack.
            pro_report = pros[pros.pro_report == 1].pro_i.unique()
            df = df[df.pro_i.isin(pro_report)]

        return df.sort_values("pep_i").reset_index(drop=True)

    def peps_above_thresholds(self, precision=0.0, recall=0.0):
        df = zap.df_groups(
            _do_peps_above_thresholds,
            self.pr_curve_by_pep().groupby("pep_i"),
            precision=precision,
            recall=recall,
            _process_mode=False,
        )
        df = df.reset_index().sort_index().rename(columns={0: "passes"})
        return np.argwhere(df.passes.values).flatten()

    def false_rates_all_peps__ptm_info(
        self,
        at_prec,
        n_false=4,
        protein_of_interest_only=True,
        ptms_column_active_only=False,
    ):
        """
        Adds some additional info requested by Angela.  I'm placing this in a separate fn
        because this is really ad-hoc info for the way we're doing PTMs at the moment and
        doesn't really belong in a more generic "false_rates..." call.
        """

        df = self.false_rates_all_peps__flus(at_prec, n_false, protein_of_interest_only)

        #
        # Add global PTM locations that occur for each peptide
        #
        pros = self._prep_result.pros()
        df = pd.merge(
            df,
            pros[["pro_i", "pro_ptm_locs"]].rename(columns=dict(pro_ptm_locs="ptms")),
            on="pro_i",
            how="left",
        )

        def ptms_in_peptide(row, only_active_ptms=ptms_column_active_only):
            # set ptms to global ptms that fall into this peptide and are active.
            # ptms are 1-based but start/stop are 0-based.
            local_ptm_indices = [
                int(i) - (row.pep_start + 1)
                for i in row.ptms.split(";")
                if i and int(i) in range(row.pep_start + 1, row.pep_stop + 1)
            ]
            if not local_ptm_indices:
                return ""
            aas = aa_str_to_list(row.seqstr)
            return ";".join(
                [
                    str(i + row.pep_start + 1)
                    for i in local_ptm_indices
                    if not only_active_ptms or "[" in aas[i]
                ]
            )

        df["ptms"] = df.apply(ptms_in_peptide, axis=1)

        #
        # Add column for "Proline in 2nd position"
        #
        df["P2"] = df.apply(
            lambda row: True
            if row.seqstr
            and len(row.seqstr) > 1
            and aa_str_to_list(row.seqstr)[1] == "P"
            else False,
            axis=1,
        )

        #
        # Add seqlen column
        #
        df["seqlen"] = df.apply(lambda row: len(aa_str_to_list(row.seqstr)), axis=1,)

        return df

    def peps__pepstrs__flustrs__p2(
        self,
        include_decoys=False,
        in_report_only=False,
        ptm_peps_only=False,
        ptms_to_rows=True,
    ):
        """
        This is collects a variety of information for reporting and is fairly configurable, thus
        the options.  How else to support these options?  The pattern of a function per join-type
        would create lots of functions in this case... Maybe only a few are needed though.
        """
        peps = self._prep_result.peps__ptms(
            include_decoys=include_decoys,
            in_report_only=in_report_only,
            ptm_peps_only=ptm_peps_only,
            ptms_to_rows=ptms_to_rows,
        )
        pepstrs = self._prep_result.pepstrs()
        flus = self._sim_result.flus()
        if "n_dyes_max_any_ch" not in flus.columns:
            # I've added this to sim but compute on demand if not in this run.
            # Remove after it's no longer needed for older runs. tfb 8 apr 2020
            self._sim_result._generate_flu_info(self._prep_result)
            flus = self._sim_result.flus()

        flustrs = flus[["pep_i", "flustr", "flu_count", "n_dyes_max_any_ch"]]

        df = (
            peps.set_index("pep_i")
            .join(pepstrs.set_index("pep_i"), how="left")
            .join(flustrs.set_index("pep_i"), how="left")
            .reset_index()
        )

        df["P2"] = df.apply(
            lambda row: True
            if row.seqstr
            and len(row.seqstr) > 1
            and aa_str_to_list(row.seqstr)[1] == "P"
            else False,
            axis=1,
        )
        return df
