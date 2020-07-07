import numpy as np
import os
from plumbum import local
from munch import Munch
from zest import zest
from plaster.tools.utils.utils import flatten, cache
from plaster.tools.utils.utils import munch_deep_copy
from plaster.tools.utils import utils
from plaster.tools.log.log import debug


def zest_flatten():
    def it_flattens_all_levels():
        f = flatten([[1, 2, [3, 4, [5, 6]]]])
        assert f == [1, 2, 3, 4, 5, 6]

    def it_flattens_one_level():
        f = flatten([[1, 2, [3, 4, [5, 6]]], [7]], depth=1)
        assert f == [1, 2, [3, 4, [5, 6]], 7]

    def it_flattens_two_levels():
        f = flatten([[1, 2, [3, 4, [5, 6]]], [7]], depth=2)
        assert f == [1, 2, 3, 4, [5, 6], 7]

    zest()


def zest_munch_deep_copy():
    src = Munch(
        some_int=1,
        some_list=[Munch(a=1, b=1.1, c="one"), Munch(a=2, b=2.1, c="two")],
        some_dict=dict(d=1, e=2, f=None,),
    )

    class Foo(Munch):
        pass

    class Bar:
        pass

    def it_makes_a_deep_copy():
        dst = munch_deep_copy(src)
        dst.some_list[0].a = 100
        assert dst.some_list[0].a == 100
        assert src.some_list[0].a == 1
        assert src.some_dict["f"] is None

    def it_raises_on_unknown_root():
        with zest.raises(TypeError):
            dst = munch_deep_copy(1)

    def it_raises_on_unknown_type():
        with zest.raises(TypeError):
            munch_deep_copy([Bar()])

    def it_converts_klasses():
        src = Munch(foo=Foo(a=1, b=2), c=3)
        dst = munch_deep_copy(src, klass_set={Foo})
        assert dst.c == 3
        dst.foo.a = 100
        assert dst.foo.a == 100
        assert src.foo.a == 1
        assert isinstance(src.foo, Foo)
        assert isinstance(dst.foo, Foo)

    zest()


def zest_cache():
    folder = local.path(local.env["ERISYON_TMP"]) / "cache"

    called = 0

    @cache()
    def __test_cacheme(a, b):
        nonlocal called
        called += 1
        return f"a={a} b={b}"

    def it_caches_to_correct_place():
        nonlocal called

        cache_glob = "cache_wrapper___test_cacheme*"

        for f in folder // cache_glob:
            f.delete()

        assert len(list(folder // cache_glob)) == 0

        called = 0
        __test_cacheme(1, 2)
        assert len(list(folder // cache_glob)) == 1
        assert called == 1

        called = 0
        __test_cacheme(1, 2)
        assert len(list(folder // cache_glob)) == 1
        assert called == 0

    zest()


class PropertiesClass:
    def __init__(self):
        self.a = "123"
        self.b = "456"
        self.c = "789"


def zest_utils_misc():
    def it_returns_value_on_in_bound_get():
        assert utils.safe_list_get(["a", "b"], 1, "bad") == "b"

    def it_returns_default_on_bad_list_get():
        assert utils.safe_list_get(["a", "b"], 2, "bad") == "bad"

    def it_returns_default_on_none_list():
        assert utils.safe_list_get(None, 2, "bad") == "bad"

    def it_flattens_a_multilevel_list():
        assert flatten([[1, 2], [3, 4, [5, "abc"]]]) == [1, 2, 3, 4, 5, "abc"]

    def it_flattens_a_tuples():
        assert flatten([[1, 2], [3, 4, (5, 6)]]) == [1, 2, 3, 4, 5, 6]

    def it_samples_with_parameters_in_order():
        calls = []

        def _trial(a, b, c):
            nonlocal calls
            calls += [(a, b, c)]
            return 123

        results = utils.sample(
            (
                # peak_std
                lambda size: np.random.uniform(1.0, 2.0, size=size),
                # hat_rad
                lambda size: np.random.randint(1, 3, size=size),
                # brim_rad
                lambda size: np.random.randint(1, 3, size=size),
            ),
            _trial,
            n_trials=2,
        )

        assert results[0] == (123, calls[0][0], calls[0][1], calls[0][2])
        assert results[1] == (123, calls[1][0], calls[1][1], calls[1][2])

    def it_creates_an_indexed_pickle_and_reloads_it():
        path = "/tmp/__test.ipkl"
        prop_list = ["a", "b"]
        testme_orig = PropertiesClass()
        utils.indexed_pickler_dump(testme_orig, path, prop_list)
        testme_new = PropertiesClass.__new__(PropertiesClass)
        utils.indexed_pickler_load(path, prop_list, testme_new)
        for prop in prop_list:
            assert getattr(testme_orig, prop) == getattr(testme_new, prop)
        assert hasattr(testme_orig, "c")
        assert not hasattr(testme_new, "c")
        os.unlink(path)

    def it_creates_an_indexed_pickle_and_reloads_it_on_all_properties():
        path = "/tmp/__test.ipkl"
        testme_orig = PropertiesClass()
        utils.indexed_pickler_dump(testme_orig, path)
        testme_new = PropertiesClass.__new__(PropertiesClass)
        utils.indexed_pickler_load(path, None, testme_new)
        for prop in ["a", "b", "c"]:
            assert getattr(testme_orig, prop) == getattr(testme_new, prop)
        os.unlink(path)

    def it_uses_two_digit_elapsed_time():
        assert utils.elapsed_time_in_minutes_seconds(0) == "0:00"
        assert utils.elapsed_time_in_minutes_seconds(600) == "10:00"

    def it_returns_non_none():
        assert utils.non_none(1, 2) == 1
        assert utils.non_none(None, 1, 2) == 1
        assert utils.non_none(None, None) is None

    def it_raises_if_none_found():
        with zest.raises(ValueError):
            assert utils.non_none(None, None, raise_if_all_none=ValueError("fail!"))

    zest()


def zest_array_same():
    def it_matches_nan():
        assert utils.np_array_same([1, np.nan], [1, np.nan])

    def it_matches_epsilon():
        assert utils.np_array_same([1 + np.finfo(np.float32).eps, np.nan], [1, np.nan])

    def it_matches_beyond_100_epsilon():
        assert not utils.np_array_same(
            [1 + 100 * np.finfo(np.float32).eps, np.nan], [1, np.nan]
        )

    zest()


def repl_1d_vecs():
    expected = np.array([[1, 1], [2, 2], [3, 3]])

    def it_repl_1d_vecs():
        r = utils.repl_vec_over_cols(np.array([1, 2, 3]), 2)
        assert utils.np_array_same(r, expected)

    def it_repl_2d_vecs_1_row():
        r = utils.repl_vec_over_cols(np.array([[1, 2, 3]]), 2)
        assert utils.np_array_same(r, expected)

    def it_repl_2d_vecs_1_col():
        r = utils.repl_vec_over_cols(np.array([[1, 2, 3]]).T, 2)
        assert utils.np_array_same(r, expected)

    def it_raises_on_non_vector():
        with zest.raises(TypeError):
            utils.repl_vec_over_cols(np.array([[1, 2], [1, 2]]), 2)

        with zest.raises(TypeError):
            utils.repl_vec_over_cols(np.array([[[1, 2], [1, 2]]]), 2)

    zest()


def block():
    def it_searches_blocks():
        assert utils.block_search({"a": [{"b": 1}, {"c": 2}]}, "a.1.c") == 2

    def it_returns_whole_block():
        assert utils.block_search({"a": [{"b": 1}, {"c": 2}]}, None) == {
            "a": [{"b": 1}, {"c": 2}]
        }

    def it_returns_pointer():
        c = {"c": 2}
        assert utils.block_search({"a": [{"b": 1}, c]}, "a.1") is c

    def it_returns_none():
        assert utils.block_search({"a": [{"b": 1}, {"c": 2}]}, "a.0.q") is None

    def it_does_not_raise_on_bad_form():
        assert utils.block_search({"a": []}, "a.q") is None

    def it_overwrites_existing_scalar():
        a = {"a": [{"b": 1}, {"c": 2}]}
        utils.block_update(a, "a.1.c", 3)
        assert a["a"][1]["c"] == 3

    def it_overwrites_existing_list_block():
        a = {"a": [{"b": 1}, {"c": 2}]}
        utils.block_update(a, "a.1", 3)
        assert a["a"][1] == 3

    def it_overwrites_existing_dict_block():
        a = {"a": [{"b": 1}, {"c": 2}]}
        utils.block_update(a, "a", 3)
        assert a["a"] == 3

    def it_adds_blocks_overwriting_scalar():
        a = {"a": [{"b": 1}, {"c": 2}]}
        utils.block_update(a, "a.0.b.d.f", 10)
        assert a["a"][0]["b"]["d"]["f"] == 10

    def it_adds_blocks_that_do_not_exist():
        a = {"sim": {"parameters": {"donotkill": 1}}}
        utils.block_update(a, "sim.parameters.peptides", [1, 2])
        assert a["sim"]["parameters"]["donotkill"] == 1
        assert a["sim"]["parameters"]["peptides"][0] == 1
        assert a["sim"]["parameters"]["peptides"][1] == 2

    def it_returns_last_where():
        a = np.array([1, 2, 2, 0, 2, 3, 3, 3])
        assert utils.np_arg_last_where(a == 2) == 4
        assert utils.np_arg_last_where(a == 3) == 7
        assert utils.np_arg_last_where(a > 3) is None
        assert utils.np_arg_last_where(a == 1) == 0
        assert utils.np_arg_last_where(np.array([])) is None

    def it_returns_none_on_empty_last_where():
        a = np.array([])
        assert utils.np_arg_last_where(a > 3) is None

    def it_safe_divides_to_zero():
        val = utils.np_safe_divide(np.array([1, 2]), np.array([0, 2]))
        assert np.all(val == [0.0, 1.0])

    def it_safe_divides_to_other_default():
        val = utils.np_safe_divide(np.array([1, 2]), np.array([0, 2]), default=10.0)
        assert np.all(val == [10.0, 1.0])

    def it_finds_keys():
        test_dict = {
            "sim": {"parameters": {"donotkill": 1}},
            "some_list": [dict(a=1, b=2), dict(b=3)],
        }
        keys = utils.block_all_keys(test_dict)
        assert keys == [
            "sim",
            "sim.parameters",
            "sim.parameters.donotkill",
            "some_list",
            "some_list.0",
            "some_list.0.a",
            "some_list.0.b",
            "some_list.1",
            "some_list.1.b",
        ]

    zest()


def zest_block_all_key_vals():
    def it_walks_dict():
        m = Munch(a=1, b=2)
        got = utils.block_all_key_vals(m)
        assert got == [("a", 1), ("b", 2)]

    def it_walks_list():
        m = [4, 3]
        got = utils.block_all_key_vals(m)
        assert got == [("0", 4), ("1", 3)]

    def it_walks_recursive():
        m = Munch(a=1, b=dict(b1=3, b2=4), c=[5, 6])
        got = utils.block_all_key_vals(m)
        assert got == [("a", 1), ("b.b1", 3), ("b.b2", 4), ("c.0", 5), ("c.1", 6)]

    zest()


def escapes():
    def it_works_on_unquoted():
        assert utils.escape_single_quotes("Hello") == "Hello"
        assert utils.escape_double_quotes("Hello") == "Hello"

    def it_works_on_single_quotes():
        assert utils.escape_single_quotes("ABC's") == "ABC\\'s"
        assert utils.escape_single_quotes("A\"BC's") == "A\"BC\\'s"

    def it_works_on_double_quotes():
        assert utils.escape_double_quotes('ABC"s') == 'ABC\\"s'
        assert utils.escape_double_quotes("A'BC\"s") == "A'BC\\\"s"

    def it_works_on_spaces():
        assert utils.escape_spaces("Hello") == "Hello"
        assert utils.escape_spaces("A B C") == "A\\ B\\ C"

    zest()


def zest_np_1d_end_pad():
    a = None

    def _before():
        nonlocal a
        a = np.array([1, 2, 3])

    def it_pads_an_array():
        assert utils.np_array_same(utils.np_1d_end_pad(a, 4), np.array([1, 2, 3, 0]))

    def it_raises_if_the_array_is_longer_than_full_size():
        with zest.raises(AssertionError):
            utils.np_1d_end_pad(a, 2)

    def it_raises_if_the_array_is_not_one_dimensional():
        with zest.raises(AssertionError):
            utils.np_1d_end_pad(np.array([[0, 1], [2, 3]]), 2)

    zest()


def zest_smart_wrap():
    def it_keeps_blank_lines():
        l = utils.smart_wrap(
            """
            ABC

            DEF
            """
        )
        assert l == "\nABC\n\nDEF\n"

    def it_keeps_indents():
        l = utils.smart_wrap(
            """
            ABC

            DEF
                GHI
            JKL
            """
        )
        assert l == "\nABC\n\nDEF\n    GHI\nJKL\n"

    zest()
