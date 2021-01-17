from concurrent.futures.process import BrokenProcessPool
import time
import numpy as np
import pandas as pd
from munch import Munch
from plaster.tools.utils.utils import listi
from plaster.tools.zap import zap
from zest import MockFunction, zest
from plaster.tools.log.log import debug


def test1(a, b, c):
    return a + b + c


def test2(a, b, c):
    raise ValueError


def _do_inner_worker(expected_mode):
    assert zap._mode == expected_mode
    time.sleep(0.1)


def _do_outer_worker(expected_mode):
    zap.work_orders(
        [dict(fn=_do_inner_worker, expected_mode=expected_mode) for task in range(10)]
    )


def zest_context():
    def it_prevents_inner_parallelism_by_default():
        assert zap._mode == "process"
        zap.work_orders(
            [dict(fn=_do_outer_worker, expected_mode="debug") for _ in range(20)]
        )

    def it_allows_inner_parallelism():
        assert zap._mode == "process"
        with zap.Context(allow_inner_parallelism=True):
            zap.work_orders(
                [dict(fn=_do_outer_worker, expected_mode="thread") for _ in range(20)]
            )

    zest()


def zest_zap_work_orders():
    # Each of the following is run in the three modes: thread, process, debug
    def _it_runs_serially(mode, work_orders):
        with zap.Context(mode=mode):
            results = zap.work_orders(work_orders)
            assert results[0] == 1 + 2 + 3
            assert results[1] == 3 + 4 + 5

    def _it_traps_exceptions_by_default(mode, work_orders):
        with zap.Context(mode=mode):
            work_orders[0].fn = test2
            results = zap.work_orders(work_orders)
            assert isinstance(results[0], ValueError)
            assert results[1] == 3 + 4 + 5

    def _it_bubbles_exceptions(mode, work_orders):
        with zest.mock(zap._show_work_order_exception) as m_ex:
            with zest.raises(ValueError):
                with zap.Context(mode=mode, trap_exceptions=False):
                    work_orders[0].fn = test2
                    zap.work_orders(work_orders)
        assert m_ex.called_once()

    def _it_calls_progress(mode, work_orders):
        progress = MockFunction()

        work_orders[0].fn = test2
        with zap.Context(mode=mode, progress=progress):
            zap.work_orders(work_orders)

        assert progress.calls == [
            ((1, 2, False), {}),
            ((2, 2, False), {}),
        ]

    def it_runs_in_debug_mode():
        work_orders = None

        def _before():
            nonlocal work_orders
            work_orders = [
                Munch(fn=test1, args=(1, 2), c=3),
                Munch(fn=test1, args=(3, 4), c=5),
            ]

        def it_runs_serially():
            _it_runs_serially("debug", work_orders)

        def it_traps_exceptions_by_default():
            _it_traps_exceptions_by_default("debug", work_orders)

        def it_bubbles_exceptions():
            _it_bubbles_exceptions("debug", work_orders)

        def it_calls_progress():
            _it_calls_progress("debug", work_orders)

        zest()

    def it_runs_in_process_mode():
        work_orders = None

        def _before():
            nonlocal work_orders
            work_orders = [
                Munch(fn=test1, args=(1, 2), c=3),
                Munch(fn=test1, args=(3, 4), c=5),
            ]

        def it_runs_serially():
            _it_runs_serially("process", work_orders)

        def it_traps_exceptions_by_default():
            _it_traps_exceptions_by_default("process", work_orders)

        def it_bubbles_exceptions():
            _it_bubbles_exceptions("process", work_orders)

        def it_calls_progress():
            _it_calls_progress("process", work_orders)

        def it_retries():
            progress = MockFunction()
            with zest.mock(zap._mock_BrokenProcessPool_exception) as m:
                m.exceptions(BrokenProcessPool)

                with zap.Context(mode="process", progress=progress):
                    results = zap.work_orders(work_orders)

                    # fmt: off
                    assert (
                        progress.calls == [((1, 2, True), {}), ((2, 2, True), {})]
                        or progress.calls == [((2, 2, True), {}), ((1, 2, True), {})]
                    )
                    # fmt: on

        zest()

    def it_runs_in_thread_mode():
        work_orders = None

        def _before():
            nonlocal work_orders
            work_orders = [
                Munch(fn=test1, args=(1, 2), c=3),
                Munch(fn=test1, args=(3, 4), c=5),
            ]

        def it_runs_serially():
            _it_runs_serially("thread", work_orders)

        def it_traps_exceptions_by_default():
            _it_traps_exceptions_by_default("thread", work_orders)

        def it_bubbles_exceptions():
            _it_bubbles_exceptions("thread", work_orders)

        def it_calls_progress():
            _it_calls_progress("thread", work_orders)

        zest()

    zest()


def test3(a, b, c):
    return [a * 2, b * 2, c * 2]


def test4(a, b, c):
    return a + 1, b + 2


def test5(a, b, c):
    return np.array([a * 2, b * 2, c * 2])


def test6(a, b, c):
    return np.array([a * 2, b * 2, c * 2]), "foo"


def zest_zap_array():
    def it_eliminates_batch_lists():
        res = zap.arrays(test3, dict(a=[1, 2], b=[3, 4]), c=3, _batch_size=2,)

        assert isinstance(res, list)
        assert res == [
            [2 * 1, 2 * 3, 2 * 3],
            [2 * 2, 2 * 4, 2 * 3],
        ]

    def it_maintains_returned_tuples():
        res = zap.arrays(test4, dict(a=[1, 2], b=[3, 4]), c=3, _batch_size=2,)

        assert isinstance(res, tuple)
        assert res == ([1 + 1, 2 + 1], [3 + 2, 4 + 2])

    def it_maintains_array_returns():
        res = zap.arrays(test5, dict(a=[1, 2], b=[3, 4]), c=3, _batch_size=2,)

        assert isinstance(res, list)
        assert np.all(res[0] == np.array([2 * 1, 2 * 3, 2 * 3]))
        assert np.all(res[1] == np.array([2 * 2, 2 * 4, 2 * 3]))

    def it_stacks_one_field():
        res = zap.arrays(
            test5, dict(a=[1, 2], b=[3, 4]), c=3, _batch_size=2, _stack=True
        )

        assert isinstance(res, np.ndarray)
        assert np.all(res == np.array([[2 * 1, 2 * 3, 2 * 3], [2 * 2, 2 * 4, 2 * 3]]))

    def it_stacks_all_fields():
        res = zap.arrays(
            test4, dict(a=[1, 2], b=[3, 4]), c=3, _batch_size=2, _stack=True
        )

        assert isinstance(res, tuple)
        assert isinstance(res[0], np.ndarray)
        assert isinstance(res[1], np.ndarray)
        assert np.all(res[0] == np.array([[1 + 1, 2 + 1]]))
        assert np.all(res[1] == np.array([[3 + 2, 4 + 2]]))

    def it_stacks_some_fields():
        res = zap.arrays(
            test6, dict(a=[1, 2], b=[3, 4]), c=3, _batch_size=2, _stack=[True, False]
        )

        assert isinstance(res, tuple)
        assert isinstance(res[0], np.ndarray)
        assert isinstance(res[1], list)
        assert np.all(
            res[0] == np.array([[1 * 2, 3 * 2, 3 * 2], [2 * 2, 4 * 2, 3 * 2]])
        )
        assert res[1] == ["foo", "foo"]

    def it_limits_slices():
        res_a, res_b = zap.arrays(
            test4,
            dict(a=np.arange(10), b=np.arange(10)),
            c=3,
            _batch_size=2,
            _limit_slice=slice(3, 6),
        )
        assert len(res_a) == 3 and len(res_b) == 3

    def it_limits_slices_with_int():
        res_a, res_b = zap.arrays(
            test6,
            dict(a=np.arange(10), b=np.arange(10)),
            c=3,
            _batch_size=2,
            _limit_slice=3,
        )
        assert len(res_a) == 3 and len(res_b) == 3

    zest()


def zest_make_batch_slices():
    def it_solves_for_batch_size_by_scaling_the_cpu_count():
        with zest.mock(zap._cpu_count, returns=2):
            sl = zap.make_batch_slices(_batch_size=None, n_rows=32, _limit_slice=None)
            assert sl == [
                (0, 3),
                (3, 6),
                (6, 9),
                (9, 12),
                (12, 15),
                (15, 18),
                (18, 21),
                (21, 24),
                (24, 27),
                (27, 30),
                (30, 32),
            ]

    def it_solves_for_batch_size_by_scaling_the_cpu_count_and_clamps():
        with zest.mock(zap._cpu_count, returns=2):
            sl = zap.make_batch_slices(_batch_size=None, n_rows=6, _limit_slice=None)
            assert sl == [(0, 2), (2, 4), (4, 6)]

    def it_uses_batch_size():
        with zest.mock(zap._cpu_count) as m:
            sl = zap.make_batch_slices(_batch_size=2, n_rows=6, _limit_slice=None)
            assert sl == [(0, 2), (2, 4), (4, 6)]
        assert not m.called()

    def it_handles_odd():
        sl = zap.make_batch_slices(_batch_size=5, n_rows=6, _limit_slice=None)
        assert sl == [(0, 5), (5, 6)]

    def it_handles_one_large_batch():
        sl = zap.make_batch_slices(_batch_size=10, n_rows=6, _limit_slice=None)
        assert sl == [(0, 6)]

    def it_handles_zero_rows():
        sl = zap.make_batch_slices(_batch_size=10, n_rows=0, _limit_slice=None)
        assert sl == []

    def it_raises_on_illegal_batch_size():
        with zest.raises(ValueError):
            zap.make_batch_slices(_batch_size=-5, n_rows=10, _limit_slice=None)

    def it_limits_from_start():
        sl = zap.make_batch_slices(_batch_size=2, n_rows=6, _limit_slice=slice(2, 6))
        assert sl == [
            (2, 4),
            (4, 6),
        ]

    zest()


def test7(row, c):
    return row.a + row.b + c, row.a * row.b * c


def test8(row, c):
    return pd.DataFrame(dict(sum_=row.a + row.b + c, prod_=row.a * row.b * c))


def zest_zap_df_rows():
    def it_raises_if_not_a_df_return():
        with zest.raises(TypeError):
            df = pd.DataFrame(dict(a=[1, 2], b=[3, 4]))
            with zap.Context(mode="debug"):
                zap.df_rows(test7, df, c=3, _batch_size=2)

    def it_splits_a_df_and_returns_a_df():
        df = pd.DataFrame(dict(a=[1, 2], b=[3, 4]))
        with zap.Context(mode="debug"):
            res = zap.df_rows(test8, df, c=3, _batch_size=2)

        assert res.equals(
            pd.DataFrame(
                [[1 + 3 + 3, 1 * 3 * 3], [2 + 4 + 3, 2 * 4 * 3],],
                columns=["sum_", "prod_"],
            )
        )

    zest()


def test9(g):
    return g.a.unique()[0], g.a.unique()[0] + 1


def zest_zap_df_groups():
    def it_groups():
        df = pd.DataFrame(dict(a=[1, 1, 2, 2, 2], b=[1, 2, 3, 4, 5]))
        res = zap.df_groups(test9, df.groupby("a"))
        a = listi(res, 0)
        ap1 = listi(res, 1)
        assert a == [1, 2]
        assert ap1 == [2, 3]

    zest()
