from munch import Munch
import numpy as np
import pandas as pd
from zest import zest
from plaster.tools.utils import utils
from plaster.run.call_bag import CallBag


from plaster.tools.log.log import debug


def zest_prs_at_prec():
    def it_computes_prs_no_ties():
        p = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = CallBag._prs_at_prec(0.2, p, r, s)
        assert result == (0.2, 0.4, 0.6)

    def it_computes_prs_with_ties():
        p = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        s = np.array([0.9, 0.6, 0.6, 0.6, 0.5])
        result = CallBag._prs_at_prec(0.2, p, r, s)
        assert result == (0.2, 0.2, 0.6)

    zest()


def zest_pr_curve_edge_cases():
    """
    Testing situations in which calls are all wrong or all right.
    """

    # first entry is null peptide
    stub_sim_result = Munch(train_pep_recalls=np.array([-1.0, 1.0, 1.0]))
    stub_prep_result = Munch(n_peps=3)

    # CallBag: all right / all wrong
    true_pep_iz = [1] * 10
    pred_pep_iz_1 = [1] * 10
    pred_pep_iz_2 = [2] * 10
    scores = [0.5] * 10
    cb_all_right = CallBag(
        sim_result=stub_sim_result,
        prep_result=stub_prep_result,
        true_pep_iz=true_pep_iz,
        pred_pep_iz=pred_pep_iz_1,
        scores=scores,
    )
    cb_all_wrong = CallBag(
        sim_result=stub_sim_result,
        prep_result=stub_prep_result,
        true_pep_iz=true_pep_iz,
        pred_pep_iz=pred_pep_iz_2,
        scores=scores,
    )

    zero_pr_result = [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]
    one_pr_result = [[1.0, 1.0], [1.0, 1.0], [0.5, 0.0], [1.0, 1.0]]

    def it_computes_zero_pr():
        p, r, s, a = cb_all_wrong.pr_curve(n_steps=2)
        assert utils.np_array_same([p, r, s, a], zero_pr_result)

    def it_computes_zero_pr_for_subset():
        p, r, s, a = cb_all_wrong.pr_curve(pep_iz_subset=[1], n_steps=2)
        assert utils.np_array_same([p, r, s, a], zero_pr_result)

        p, r, s, a = cb_all_wrong.pr_curve(pep_iz_subset=[2], n_steps=2)
        assert utils.np_array_same([p, r, s, a], zero_pr_result)

        # peptide 2 does not show up in true/pred at all so should get zero pr curve
        p, r, s, a = cb_all_right.pr_curve(pep_iz_subset=[2], n_steps=2)
        assert utils.np_array_same([p, r, s, a], zero_pr_result)

    def it_computes_one_pr():
        p, r, s, a = cb_all_right.pr_curve(n_steps=2)
        # ugh, can't use utils.flatten bc elems are ndarray, not list or tuple
        compare = [a == b for a, b in zip([p, r, s, a], one_pr_result)]
        compare = [list(el) if type(el) is np.ndarray else el for el in compare]
        assert all(compare)

    def it_computes_one_pr_for_subset():
        p, r, s, a = cb_all_right.pr_curve(pep_iz_subset=[1], n_steps=2)
        compare = [a == b for a, b in zip([p, r, s, a], one_pr_result)]
        compare = [list(el) if type(el) is np.ndarray else el for el in compare]
        assert all(compare)

    def it_handles_all_rows_no_recall():
        p, r, s, a = cb_all_wrong.pr_curve()
        assert np.all(r == 0.0)

    zest()


def zest_pr_curve_no_tied_scores():
    """
    Testing situations with some right and some wrong calls, but no tied scores.
    """

    # first entry is null peptide
    stub_sim_result = Munch(train_pep_recalls=np.array([-1.0] + [1.0] * 4))
    stub_prep_result = Munch(n_peps=5)

    # pep 1 is predicted correctly 1/4, #2 is 1/2, #3 is 3/4
    true_pep_iz = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    pred_pep_iz = np.array([1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1])
    scores = np.array(
        [0.8, 0.9, 0.7, 0.6, 0.85, 0.53, 0.54, 0.55, 0.75, 0.4, 0.3, 0.35]
    )

    cb = CallBag(
        sim_result=stub_sim_result,
        prep_result=stub_prep_result,
        true_pep_iz=true_pep_iz,
        pred_pep_iz=pred_pep_iz,
        scores=scores,
    )

    # sorted_i = np.argsort( scores )[::-1]

    # sorted by score, highest->lowest
    # t[1.  2.   1.  3.   1.  1.  2.   2.   2.   3.  3.   3.]
    # p[2.  2.   1.  3.   2.  2.  3.   3.   2.   3.  1.   3.]
    # s[0.9 0.85 0.8 0.75 0.7 0.6 0.55 0.54 0.53 0.4 0.35 0.3]

    # cumulative sum of correct calls, cumulative call count
    # [F.  T.   T.  T.   F.  F.  F.   F.   T.   T.  F.  T.]
    # cum_sum_correct = np.array(
    #     [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0]
    # )
    # cum_sum_count = np.array(range(1, len(pred_pep_iz) + 1))

    # Now using a linear stepper of the score so these are now:
    # [F.   T.   T.   T.   F.   F.   F.   F.   T.   T.   F.   T.]
    # [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0]
    #  .9        .8        .7   .6             .5   .4        .3

    # Note that the new pr_curve trims out starting values of r=0
    # So the p=0.0 and r=0./12. are removed
    prec = np.array(
        [
            # 0.0 / 1.0,  # 0.9
            2.0 / 3.0,  # 0.8
            3.0 / 5.0,  # 0.7
            3.0 / 6.0,  # 0.6
            4.0 / 9.0,  # 0.5
            5.0 / 10.0,  # 0.4
            6.0 / 12.0,  # 0.3
            6.0 / 12.0,  # 0.2
            6.0 / 12.0,  # 0.1
            6.0 / 12.0,  # 0.0
        ]
    )
    reca = np.array(
        [
            # 0.0 / 12.0,  # 0.9
            2.0 / 12.0,  # 0.8
            3.0 / 12.0,  # 0.7
            3.0 / 12.0,  # 0.6
            4.0 / 12.0,  # 0.5
            5.0 / 12.0,  # 0.4
            6.0 / 12.0,  # 0.3
            6.0 / 12.0,  # 0.2
            6.0 / 12.0,  # 0.1
            6.0 / 12.0,  # 0.0
        ]
    )

    # precision at score threshold
    # prec_at_thresh = utils.np_safe_divide(cum_sum_correct, cum_sum_count, default=0)
    # prec_at_thresh = np.append([1.0], prec_at_thresh)

    # recall at each threshold
    # recall_at_thresh = cum_sum_correct / len(pred_pep_iz)
    # recall_at_thresh = np.append([0.0], recall_at_thresh)

    def it_computes_combined_pr():
        p, r, s, a = cb.pr_curve(n_steps=10)
        assert np.array_equal(p, prec)
        assert np.allclose(s, np.linspace(0.8, 0.0, 9))
        assert np.array_equal(r, reca)

    def it_computes_subset_pr():
        p, r, s, a = cb.pr_curve(pep_iz_subset=[1], n_steps=10)

        # t[1.       1.       1.  1.                     3.     ]
        # p[2.       1.       2.  2.                     1.     ]
        # s[0.9 0.85 0.8 0.75 0.7 0.6 0.55 0.54 0.53 0.4 0.35 0.3]

        prec = np.array(
            [
                # 0.0 / 1.0,  # 0.9
                1.0 / 1.0,  # 0.8
                1.0 / 1.0,  # 0.7
                1.0 / 1.0,  # 0.6
                1.0 / 1.0,  # 0.5
                1.0 / 1.0,  # 0.4
                1.0 / 2.0,  # 0.3
                1.0 / 2.0,  # 0.2
                1.0 / 2.0,  # 0.1
                1.0 / 2.0,  # 0.0
            ]
        )
        reca = np.array(
            [
                # 0.0 / 1.0,  # 0.9
                1.0 / 4.0,  # 0.8
                1.0 / 4.0,  # 0.7
                1.0 / 4.0,  # 0.6
                1.0 / 4.0,  # 0.5
                1.0 / 4.0,  # 0.4
                1.0 / 4.0,  # 0.3
                1.0 / 4.0,  # 0.2
                1.0 / 4.0,  # 0.1
                1.0 / 4.0,  # 0.0
            ]
        )

        # look at true & pred that contain 1
        #          t[1.  1.  1.  1.  3.  ]
        #          p[2.  1.  2.  2.  1.  ]
        #          s[0.9 0.8 0.7 0.6 0.35]
        #
        # calculated prec first, based only on predictions
        # [1] out front is threshold added by sklearn
        # prec = [1][    1,         1/2]
        #
        # the scores for those are used as thresholds:
        # scor = [1][     8,          35 ]

        # the recalls are found by looking at trues at threshold scores
        # e.g. "what fraction of trues have been successfully found at score threshold .8?"
        # reca = [0][    .25,        .25]

        # Note that sklearn truncates when full recall has been obtained,
        # so we'll lose the last element of each of the above since max
        # recall occurs at recall=0.25 -- last one is dropped.  Said another
        # way: "The number of true positives does not go up after the one found
        # at score = 0.8, so stop reporting precision/recall/thresholds right there."
        assert np.array_equal(p, prec)
        assert np.array_equal(r, reca)
        assert np.allclose(s, np.linspace(0.8, 0.0, 9))

        #
        # Test again for a different subset.
        #
        p, r, s, a = cb.pr_curve(pep_iz_subset=[2], n_steps=10)
        #          t[1.  2.    1.  1.  2.   2.   2.  ]
        #          p[2.  2.    2.  2.  3.   3.   2.  ]
        #          s[0.9 0.85  0.7 0.6 0.55 0.54 0.53]
        #
        # calculated prec first, based only on predictions
        # prec = [1][0, .5,   1/3, 1/4,          2/5 ]
        #
        # the scores for those are used as thresholds:
        # scor = [1][9, 85,   7,   6,             53  ]

        # the recalls are found by looking trues and thresholds
        # reca = [0][0, 1/4,  1/4, 1/4,           2/4  ]

        # t[ 1.  2.              1.   1.  2.   2.    2.              ]
        # p[ 2.  2.              2.   2.  3.   3.    2.              ]
        #  [.9        .8        .7   .6             .5   .4        .3

        prec = np.array(
            [
                # 0.0 / 2.0,  # 0.9
                1.0 / 2.0,  # 0.8
                1.0 / 3.0,  # 0.7
                1.0 / 4.0,  # 0.6
                2.0 / 5.0,  # 0.5
                2.0 / 5.0,  # 0.4
                2.0 / 5.0,  # 0.3
                2.0 / 5.0,  # 0.2
                2.0 / 5.0,  # 0.1
                2.0 / 5.0,  # 0.0
            ]
        )
        reca = np.array(
            [
                # 0.0 / 0.0,  # 0.9
                1.0 / 4.0,  # 0.8
                1.0 / 4.0,  # 0.7
                1.0 / 4.0,  # 0.6
                2.0 / 4.0,  # 0.5
                2.0 / 4.0,  # 0.4
                2.0 / 4.0,  # 0.3
                2.0 / 4.0,  # 0.2
                2.0 / 4.0,  # 0.1
                2.0 / 4.0,  # 0.0
            ]
        )

        assert np.array_equal(p, prec)
        assert np.array_equal(r, reca)
        assert np.allclose(s, np.linspace(0.8, 0.0, 9))

    zest()


def zest_pr_curve_no_tied_scores_mean_recall():
    """
    Testing situations with some right and some wrong calls, but no tied scores.
    Adding to this different train_recall factors to test that those get included
    in recall calculations.

    Same fixtures as above except the means, see above for comments.
    """

    # first entry is null peptide
    stub_sim_result = Munch(train_pep_recalls=np.array([-1.0, 0.1, 0.2, 0.3]))
    stub_prep_result = Munch(n_peps=4)

    true_pep_iz = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    pred_pep_iz = np.array([1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1])
    scores = np.array(
        [0.8, 0.9, 0.7, 0.6, 0.85, 0.53, 0.54, 0.55, 0.75, 0.4, 0.3, 0.35]
    )

    cb = CallBag(
        sim_result=stub_sim_result,
        prep_result=stub_prep_result,
        true_pep_iz=true_pep_iz,
        pred_pep_iz=pred_pep_iz,
        scores=scores,
    )

    cum_sum_correct = np.array(
        [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0]
    )
    cum_sum_count = np.array(range(1, len(pred_pep_iz) + 1))

    # precision at score threshold
    prec_at_thresh = utils.np_safe_divide(cum_sum_correct, cum_sum_count, default=0)
    prec_at_thresh = np.append([1.0], prec_at_thresh)

    # recall at each threshold
    recall_at_thresh = cum_sum_correct / len(pred_pep_iz)
    recall_at_thresh = np.append([0.0], recall_at_thresh)

    zest()
