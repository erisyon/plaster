import numpy as np
import pandas as pd
from munch import Munch
from zest import zest, MockFunction
from plaster.tools.zap import zap
from concurrent.futures.process import BrokenProcessPool
from plaster.tools.log.log import debug
from plaster.tools.utils.utils import listi


def test1(a, b, c):
    return a + b + c


def test2(a, b, c):
    raise ValueError


@zest.group("integration")
def zest_zap_work_orders():
    def it_runs_in_debug_mode():
        work_orders = None

        def _before():
            nonlocal work_orders
            work_orders = [
                Munch(fn=test1, args=(1, 2), c=3),
                Munch(fn=test1, args=(3, 4), c=5),
            ]

        def it_runs_serially():
            results = zap.work_orders(work_orders, _debug_mode=True)
            assert results[0] == 1 + 2 + 3
            assert results[1] == 3 + 4 + 5

        def it_traps_exceptions():
            work_orders[0].fn = test2
            results = zap.work_orders(
                work_orders, _debug_mode=True, _trap_exceptions=True,
            )
            assert isinstance(results[0], ValueError)
            assert results[1] == 3 + 4 + 5

        def it_bubbles_exceptions():
            with zest.mock(zap._show_work_order_exception) as m_ex:
                with zest.raises(ValueError):
                    work_orders[0].fn = test2
                    zap.work_orders(
                        work_orders, _debug_mode=True, _trap_exceptions=False,
                    )
            assert m_ex.called_once()

        def it_calls_progress():
            progress = MockFunction()

            work_orders[0].fn = test2
            zap.work_orders(
                work_orders, _debug_mode=True, _progress=progress,
            )

            assert progress.calls == [
                ((1, 2, False), {}),
                ((2, 2, False), {}),
            ]

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
            results = zap.work_orders(work_orders, _process_mode=True)
            assert results[0] == 1 + 2 + 3
            assert results[1] == 3 + 4 + 5

        def it_traps_exceptions():
            work_orders[0].fn = test2
            results = zap.work_orders(
                work_orders, _process_mode=True, _trap_exceptions=True,
            )
            assert isinstance(results[0], ValueError)
            assert results[1] == 3 + 4 + 5

        def it_bubbles_exceptions():
            with zest.mock(zap._show_work_order_exception) as m_ex:
                with zest.raises(ValueError):
                    work_orders[0].fn = test2
                    zap.work_orders(
                        work_orders, _process_mode=True, _trap_exceptions=False,
                    )
            assert m_ex.called_once()

        def it_calls_progress():
            progress = MockFunction()

            work_orders[0].fn = test2
            zap.work_orders(
                work_orders, _process_mode=True, _progress=progress,
            )

            assert progress.calls == [
                ((1, 2, False), {}),
                ((2, 2, False), {}),
            ]

        def it_retries():
            progress = MockFunction()
            with zest.mock(zap._mock_BrokenProcessPool_exception) as m:
                m.exceptions(BrokenProcessPool)

                results = zap.work_orders(
                    work_orders, _process_mode=True, _progress=progress
                )
                assert progress.calls == [
                    ((1, 2, True), {}),
                    ((2, 2, True), {}),
                ]

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
            results = zap.work_orders(work_orders, _process_mode=False)
            assert results[0] == 1 + 2 + 3
            assert results[1] == 3 + 4 + 5

        def it_traps_exceptions():
            work_orders[0].fn = test2
            results = zap.work_orders(
                work_orders, _process_mode=False, _trap_exceptions=True,
            )
            assert isinstance(results[0], ValueError)
            assert results[1] == 3 + 4 + 5

        def it_bubbles_exceptions():
            with zest.mock(zap._show_work_order_exception) as m_ex:
                with zest.raises(ValueError):
                    work_orders[0].fn = test2
                    zap.work_orders(
                        work_orders, _process_mode=False, _trap_exceptions=False,
                    )
            assert m_ex.called_once()

        def it_calls_progress():
            progress = MockFunction()

            work_orders[0].fn = test2
            zap.work_orders(
                work_orders, _process_mode=False, _progress=progress,
            )

            assert progress.calls == [
                ((1, 2, False), {}),
                ((2, 2, False), {}),
            ]

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


@zest.group("integration")
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


@zest.group("integration")
def zest_make_batch_slices():
    def it_solves_for_batch_size_by_scaling_the_cpu_count():
        with zest.mock(zap._cpu_count, returns=2):
            sl = zap._make_batch_slices(_batch_size=None, n_rows=32, _limit_slice=None)
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
            sl = zap._make_batch_slices(_batch_size=None, n_rows=6, _limit_slice=None)
            assert sl == [(0, 2), (2, 4), (4, 6)]

    def it_uses_batch_size():
        with zest.mock(zap._cpu_count) as m:
            sl = zap._make_batch_slices(_batch_size=2, n_rows=6, _limit_slice=None)
            assert sl == [(0, 2), (2, 4), (4, 6)]
        assert not m.called()

    def it_handles_odd():
        sl = zap._make_batch_slices(_batch_size=5, n_rows=6, _limit_slice=None)
        assert sl == [(0, 5), (5, 6)]

    def it_handles_one_large_batch():
        sl = zap._make_batch_slices(_batch_size=10, n_rows=6, _limit_slice=None)
        assert sl == [(0, 6)]

    def it_handles_zero_rows():
        sl = zap._make_batch_slices(_batch_size=10, n_rows=0, _limit_slice=None)
        assert sl == []

    def it_raises_on_illegal_batch_size():
        with zest.raises(ValueError):
            zap._make_batch_slices(_batch_size=-5, n_rows=10, _limit_slice=None)

    def it_limits_from_start():
        sl = zap._make_batch_slices(_batch_size=2, n_rows=6, _limit_slice=slice(2, 6))
        assert sl == [
            (2, 4),
            (4, 6),
        ]

    zest()


def test7(row, c):
    return row.a + row.b + c, row.a * row.b * c


def test8(row, c):
    return pd.DataFrame(dict(sum_=row.a + row.b + c, prod_=row.a * row.b * c))


@zest.group("integration")
def zest_zap_df_rows():
    def it_raises_if_not_a_df_return():
        with zest.raises(TypeError):
            df = pd.DataFrame(dict(a=[1, 2], b=[3, 4]))
            zap.df_rows(
                test7, df, c=3, _batch_size=2, _debug_mode=True,
            )

    def it_splits_a_df_and_returns_a_df():
        df = pd.DataFrame(dict(a=[1, 2], b=[3, 4]))
        res = zap.df_rows(test8, df, c=3, _batch_size=2, _debug_mode=True,)

        assert res.equals(
            pd.DataFrame(
                [[1 + 3 + 3, 1 * 3 * 3], [2 + 4 + 3, 2 * 4 * 3],],
                columns=["sum_", "prod_"],
            )
        )

    zest()


def test9(g):
    return g.a.unique()[0], g.a.unique()[0] + 1


@zest.group("integration")
def zest_zap_df_groups():
    def it_groups():
        df = pd.DataFrame(dict(a=[1, 1, 2, 2, 2], b=[1, 2, 3, 4, 5]))
        res = zap.df_groups(test9, df.groupby("a"))
        a = listi(res, 0)
        ap1 = listi(res, 1)
        assert a == [1, 2]
        assert ap1 == [2, 3]

    zest()
