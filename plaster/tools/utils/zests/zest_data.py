import numpy as np
from zest import zest
from plaster.tools.utils.data import ConfMat, cluster, subsample, arg_subsample
from plaster.tools.log.log import debug
from plaster.tools.schema import check


def zest_data():
    a = np.array([[0, 0, 1.1], [1, 1.0, 1], [1, 1.1, 1], [0, 0.1, 1]])
    l = list("abcdef")
    assert len(l) == 6

    def it_clusters():
        def it_sorts():
            b = cluster(a)
            assert np.all(
                b == np.array([[1, 1.1, 1], [1, 1.0, 1], [0, 0.1, 1], [0, 0, 1.1]])
            )

        def it_subsamples_cluster():
            b = cluster(a, n_subsample=1)
            assert b.shape == (1, 3)

        zest()

    def it_subsamples_array():
        b = subsample(a, 2)
        assert b.shape == (2, 3)

    def it_arg_subsamples_array():
        b = arg_subsample(a, 2)
        assert b.shape == (2,)

    def it_subsamples_list():
        b = subsample(l, 2)
        assert len(b) == 2
        assert isinstance(b[0], str)

    def it_arg_subsamples_list():
        b = arg_subsample(l, 2)
        assert len(b) == 2
        assert isinstance(b[0], int)

    zest()


def zest_ConfMat():
    def it_creates_from_arrays():
        mat = np.array([[1, 2], [3, 4]])
        cm = ConfMat.from_array(mat)
        assert np.all(cm == mat)
        assert mat is not cm

    def it_creates_from_true_pred():
        true = np.array([1, 2, 3])
        pred = np.array([1, 2, 3])
        cm = ConfMat.from_true_pred(true, pred, true_dim=4, pred_dim=4)
        assert np.all(cm == [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        def it_asserts_that_no_elements_are_outside_of_dim():
            with zest.raises(AssertionError):
                ConfMat.from_true_pred(true, pred, true_dim=2, pred_dim=2)

        zest()

    def it_extracts_precision_without_div_0():
        cm = ConfMat.from_array(np.array([[5, 0], [0, 0]]))
        prec = cm.precision()
        assert np.all(prec == [1.0, 0])

    def it_extracts_recall_without_div_0():
        cm = ConfMat.from_array(np.array([[5, 0], [5, 0]]))
        rec = cm.recall()
        assert np.all(rec == [0.5, 0])

    zest()


def zest_false_calls():
    cm = ConfMat.from_array(
        np.array([[0, 0, 1, 0], [0, 9, 0, 0], [0, 2, 9, 4], [0, 0, 2, 9]])
    )

    def it_find_the_top_false():
        false_tups = cm.false_calls(elem_i=2, n_false=2)
        assert false_tups == [
            ("FP0", 3, 4.0 / 15.0),
            ("FP1", 1, 2.0 / 15.0),
            ("FN0", 3, 2.0 / 12.0),
        ]

    def it_validates_elem_i():
        with zest.raises(check.CheckAffirmError, in_message="elem_i out of range"):
            cm.false_calls(elem_i=10, n_false=1)

    def it_returns_none_invalid_n_false():
        cm.false_calls(elem_i=1, n_false=10) is None

    zest()
