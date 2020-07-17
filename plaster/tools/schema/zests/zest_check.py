import numpy as np
from zest import zest
from plaster.tools.schema import check


def test_func():
    some_float = 1.0
    check.t(some_float, int)


def zest_check():
    zest.stack_mock(check.log.error)

    def it_converts_none_to_type_none_scalar():
        a = None
        check.t(a, None)

    def it_converts_none_to_type_none_tuple():
        a = None
        check.t(a, (None,))

    def it_gets_var_name_and_source():
        with zest.raises(check.CheckError) as e:
            test_func()
        assert e.exception.var_name == "some_float"
        assert "zest_check.py" in e.exception.source

    def it_checks_type_tuples():
        some_float = 1.0
        some_int = 1
        check.t(some_float, (float, int))
        check.t(some_int, (float, int))

    def it_checks_lists():
        l = [1, 2, 3]
        check.list_t(l, int)

        l = [1, 2, 3, 4.0]
        with zest.raises(check.CheckError):
            check.list_t(l, int)

        t = (1, 2, 3)
        with zest.raises(check.CheckError):
            check.list_t(t, int)

    def it_checks_lists_or_tuples():
        l = [1, 2, 3]
        check.list_or_tuple_t(l, int)

        t = (1, 2, 3)
        check.list_or_tuple_t(t, int)

        l = [1, 2, 3.0]
        with zest.raises(check.CheckError):
            check.list_or_tuple_t(l, int)

        t = (1, 2, 3.0)
        with zest.raises(check.CheckError):
            check.list_or_tuple_t(t, int)

    def zest_arrays():
        arr = np.array([1, 2, 3])

        def it_checks_is_array():
            with zest.raises(check.CheckError):
                check.array_t([])

        def it_checks_is_dtype_if_specified():
            check.array_t(arr, dtype=np.int64)

            with zest.raises(check.CheckError):
                check.array_t(arr, dtype=np.float64)

        def it_prints_array_shape_if_not_specified():
            with zest.mock(check._print) as p:
                check.array_t(arr)
            assert "(3,)" in p.normalized_call()["msg"]

        zest()

    def zest_test():
        def it_passes():
            check.affirm(True)

        def it_raises_checkerror_by_default():
            with zest.raises(check.CheckAffirmError):
                check.affirm(False)

        def it_accepts_exception_type():
            with zest.raises(ValueError):
                check.affirm(False, exp=ValueError)

        def it_accepts_exception_instance():
            with zest.raises(ValueError) as e:
                check.affirm(False, exp=ValueError())

        def it_pushes_msg():
            with zest.raises(check.CheckAffirmError) as e:
                check.affirm(False, "abc")
            assert e.exception.message == "abc"

        zest()

    zest()


@check.args
def my_test_global_func(a: str, b):
    assert isinstance(a, str)


def zest_check_args():
    zest.stack_mock(check.log.error)

    def it_raises_on_bad_argument_global():
        with zest.raises(check.CheckError) as e:
            my_test_global_func(1, 2)

    def it_raises_on_bad_argument_local():
        with zest.raises(check.CheckError) as e:

            @check.args
            def myfunc(a: str, b):
                assert isinstance(a, str)

            myfunc(1, 2)

            zest()

    zest()
