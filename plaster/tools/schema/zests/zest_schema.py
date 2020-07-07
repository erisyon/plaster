import numpy as np
from zest import zest
from plaster.tools.schema.schema import Schema as s
from plaster.tools.schema.schema import SchemaInvalid, SchemaValidationFailed


def zest_schema():
    m_print_error = zest.stack_mock(s._print_error)
    m_print_help = zest.stack_mock(s._print_help)

    def _it_validates_noneable(schema_func):
        test_s = s(schema_func())
        with zest.raises(SchemaValidationFailed):
            test_s.validate(None)

        test_s = s(schema_func(noneable=True))
        test_s.validate(None)

    def it_raises_on_a_bad_schema():
        with zest.raises(SchemaInvalid):
            s(1)

    def it_prints_by_default():
        with zest.raises(SchemaValidationFailed):
            test_s = s(s.is_int())
            test_s.validate("not int")
        assert m_print_error.called()

    def it_raises_by_default():
        with zest.raises(SchemaValidationFailed):
            test_s = s(s.is_int())
            test_s.validate("not int")

    def check_bounds():
        def it_validates_min_val():
            s._check_bounds(5, bounds=(5, None))
            with zest.raises(SchemaValidationFailed):
                s._check_bounds(3, bounds=(5, None))

        def it_validates_max_val():
            s._check_bounds(5, bounds=(None, 5))
            with zest.raises(SchemaValidationFailed):
                s._check_bounds(6, bounds=(None, 5))

        def it_validates_bounds():
            s._check_bounds(4, bounds=(4, 6))
            s._check_bounds(6, bounds=(4, 6))
            with zest.raises(SchemaValidationFailed):
                s._check_bounds(3, bounds=(4, 6))
                s._check_bounds(7, bounds=(4, 6))

        def it_raises_if_bounds_not_valid():
            with zest.raises(SchemaInvalid):
                s._check_bounds_arg(bounds=4)
            with zest.raises(SchemaInvalid):
                s._check_bounds_arg(bounds=("a", "b"))
            with zest.raises(SchemaInvalid):
                s._check_bounds_arg(bounds=())

        zest()

    def types():
        def is_int():
            def it_validates_int():
                test_s = s(s.is_int())
                test_s.validate(1)
                with zest.raises(SchemaValidationFailed):
                    test_s.validate("a str")
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1.0)

            def it_validates_noneable():
                _it_validates_noneable(s.is_int)

            zest()

        def is_float():
            def it_validates_float():
                test_s = s(s.is_float())
                test_s.validate(1.0)
                with zest.raises(SchemaValidationFailed):
                    test_s.validate("a str")
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1)

            def it_validates_noneable():
                _it_validates_noneable(s.is_float)

            zest()

        def is_number():
            def it_validates_number():
                test_s = s(s.is_number())
                test_s.validate(1.0)
                test_s.validate(1)
                with zest.raises(SchemaValidationFailed):
                    test_s.validate("a str")

            def it_validates_noneable():
                _it_validates_noneable(s.is_number)

            zest()

        def is_str():
            def it_validates_str():
                test_s = s(s.is_str())
                test_s.validate("test")
                test_s.validate("")
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1)

            def it_validates_noneable():
                _it_validates_noneable(s.is_str)

            def it_validates_allow_empty_string():
                test_s = s(s.is_str(allow_empty_string=False))
                test_s.validate("test")
                with zest.raises(SchemaValidationFailed):
                    test_s.validate("")

                test_s = s(s.is_str())
                test_s.validate("test")
                test_s.validate("")

            zest()

        def is_bool():
            def it_validates_bool():
                test_s = s(s.is_bool())
                test_s.validate(True)
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1)

            def it_validates_noneable():
                _it_validates_noneable(s.is_str)

            zest()

        def is_deprecated():
            def it_validates_any_usage():
                test_s = s(s.is_deprecated())
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1)

            def it_never_requires_deprecated():
                test_s = s(s.is_kws_r(a=s.is_deprecated()))
                test_s.validate(dict())

            zest()

        def is_list():
            def it_validates_default_list():
                test_s = s(s.is_list())
                test_s.validate([])
                test_s.validate([1, 2, 3, "str", dict(), []])
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1)

            def it_has_elems_as_first_arg():
                test_s = s(s.is_list(s.is_int()))
                test_s.validate([1])
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(["str"])

            def it_validates_noneable():
                _it_validates_noneable(s.is_list)

            def it_validates_default_list_elems_int():
                test_s = s(s.is_list(elems=s.is_int()))
                test_s.validate([1, 2, 3])
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1)
                with zest.raises(SchemaValidationFailed):
                    test_s.validate([1, "str"])

            def it_checks_bound_type():
                with zest.raises(SchemaInvalid):
                    s(s.is_list(min_len="str"))
                with zest.raises(SchemaInvalid):
                    s(s.is_list(max_len="str"))

            def it_bounds_min():
                test_s = s(s.is_list(min_len=2))
                test_s.validate([1, 2])
                with zest.raises(SchemaValidationFailed):
                    test_s.validate([1])
                with zest.raises(SchemaValidationFailed):
                    test_s.validate([])

            def it_bounds_max():
                test_s = s(s.is_list(max_len=2))
                test_s.validate([])
                test_s.validate([1, 2])
                with zest.raises(SchemaValidationFailed):
                    test_s.validate([1, 2, 3])

            zest()

        def is_dict():
            def it_validates_default_dict():
                test_s = s(s.is_dict())
                test_s.validate({})
                test_s.validate(dict(a=1, b=2))
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(1)

            def it_validates_noneable():
                _it_validates_noneable(s.is_dict)

            def it_ignores_underscored_keys():
                test_s = s(
                    s.is_dict(elems=dict(a=s.is_int(), b=s.is_int()), no_extras=True)
                )
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(dict(a=1, b="str"))
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(dict(a=1, b=2, _c=[]))

                test_s = s(
                    s.is_dict(
                        elems=dict(a=s.is_int(), b=s.is_int()),
                        no_extras=True,
                        ignore_underscored_keys=True,
                    )
                )
                test_s.validate(dict(a=1, b=2, _c=[]))

            def it_all_required_false_by_default():
                test_s = s(s.is_dict(elems=dict(a=s.is_int(), b=s.is_int())))
                test_s.validate(dict(a=1))

            def it_checks_all_required():
                test_s = s(
                    s.is_dict(elems=dict(a=s.is_int(), b=s.is_int()), all_required=True)
                )
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(dict(a=1))

            def it_allows_all_required_to_be_overriden():
                test_s = s(
                    s.is_dict(elems=dict(a=s.is_int(required=False)), all_required=True)
                )
                test_s.validate(dict())

            def it_checks_no_extra_flase_by_default():
                test_s = s(s.is_dict(elems=dict(a=s.is_int(), b=s.is_int())))
                test_s.validate(dict(a=1, c=1))

            def it_checks_no_extra():
                test_s = s(
                    s.is_dict(elems=dict(a=s.is_int(), b=s.is_int()), no_extras=True)
                )
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(dict(a=1, b=1, c=1))

            def it_checks_key_type_str():
                test_s = s(s.is_dict(elems={1: s.is_int()}))
                with zest.raises(SchemaValidationFailed):
                    test_s.validate({1: 2})

            def it_checks_required():
                test_s = s(
                    s.is_dict(elems=dict(a=s.is_int(required=True), b=s.is_int()))
                )
                with zest.raises(SchemaValidationFailed):
                    test_s.validate(dict(b=1))

            zest()

        zest()

    class TestType:
        pass

    def is_type():
        def it_validates_type():
            test_s = s(s.is_type(TestType))
            test_s.validate(TestType())
            with zest.raises(SchemaValidationFailed):
                test_s.validate("a str")
            with zest.raises(SchemaValidationFailed):
                test_s.validate(1.0)

        def it_validates_noneable():
            test_s = s(s.is_type(TestType))
            with zest.raises(SchemaValidationFailed):
                test_s.validate(None)
            test_s = s(s.is_type(TestType, noneable=True))
            test_s.validate(None)

        zest()

    def it_validates_recursively():
        test_s = s(
            s.is_dict(
                elems=dict(
                    a=s.is_int(),
                    b=s.is_list(required=True, elems=s.is_str()),
                    c=s.is_dict(required=True),
                )
            )
        )
        test_s.validate(dict(a=1, b=["a", "b"], c=dict()))
        with zest.raises(SchemaValidationFailed):
            test_s.validate(dict(a=1, b=[1], c=dict()))
        with zest.raises(SchemaValidationFailed):
            test_s.validate(dict(a=1, b=["a"], c=1))

    def defaults():
        test_s = None

        def _before():
            nonlocal test_s
            test_s = s(
                s.is_dict(
                    all_required=True,
                    elems=dict(
                        a=s.is_int(),
                        b=s.is_int(),
                        c=s.is_dict(
                            all_required=True, elems=dict(d=s.is_int(), e=s.is_int())
                        ),
                    ),
                )
            )

        def it_applies_defaults_recursively():
            test = dict(a=1, c=dict(e=10))

            with zest.raises(SchemaValidationFailed):
                test_s.validate(test)

            test_s.apply_defaults(dict(a=2, b=3, c=dict(d=4)), apply_to=test)
            assert test["a"] == 1
            assert test["b"] == 3
            assert test["c"]["d"] == 4
            assert test["c"]["e"] == 10
            test_s.validate(test)

        def it_does_not_apply_defaults_to_none_by_default():
            test = dict(a=None, b=1, c=None)
            test_s.apply_defaults(dict(a=1, b=3, c=dict(d=4, e=5)), apply_to=test)
            assert test["a"] is None
            assert test["c"] is None

        def it_applies_defaults_on_none():
            test = dict(a=None, b=1, c=None)
            test_s.apply_defaults(
                dict(a=1, b=3, c=dict(d=4, e=5)), apply_to=test, override_nones=True
            )
            assert test["a"] == 1
            assert test["b"] == 1
            assert test["c"]["d"] == 4
            assert test["c"]["e"] == 5

        def it_allows_elems_in_dict():
            s(s.is_dict(dict(a=s.is_int(noneable=True))))

        def it_raises_on_a_missing_default():
            test = dict(a=1, c=dict())
            test_s.apply_defaults(defaults=dict(a=2, b=3), apply_to=test)
            with zest.raises(SchemaValidationFailed):
                test_s.validate(test)

        def it_applies_a_none_to_a_missing_key():
            test = dict()
            test_s.apply_defaults(defaults=dict(a=None), apply_to=test)
            assert test["a"] is None

        def it_applies_a_none_to_a_missing_dict():
            test = dict()
            test_s.apply_defaults(defaults=dict(c=None), apply_to=test)
            assert test["c"] is None

        def it_does_not_overwrite_an_existing_dict():
            test_s = s(s.is_kws_r(a=s.is_dict()))
            test = dict(a=dict(b=1))
            # a has a good value, do not overwrite it
            test_s.apply_defaults(defaults=dict(a=None), apply_to=test)
            assert test["a"]["b"] == 1

        zest()

    def requirements():
        def it_returns_required_elems():
            userdata = dict(some_key=1)

            test_s = s(
                s.is_dict(
                    all_required=True,
                    elems=dict(
                        a=s.is_int(),
                        b=s.is_float(help="A float"),
                        c=s.is_number(),
                        d=s.is_str(userdata=userdata),
                        e=s.is_list(),
                        f=s.is_dict(
                            all_required=True, elems=dict(d=s.is_int(), e=s.is_int())
                        ),
                    ),
                )
            )
            reqs = test_s.requirements()
            assert reqs == [
                ("a", int, None, None),
                ("b", float, "A float", None),
                ("c", float, None, None),
                ("d", str, None, userdata),
                ("e", list, None, None),
                ("f", dict, None, None),
            ]

        def it_returns_none_on_a_non_dict_schema():
            test_s = s(s.is_str())
            reqs = test_s.requirements()
            assert reqs == []

        zest()

    def it_shows_help():
        schema = s(
            s.is_kws(
                a=s.is_dict(
                    help="Help for a",
                    elems=dict(
                        b=s.is_int(help="Help for b"),
                        c=s.is_kws(d=s.is_int(help="Help for d")),
                    ),
                )
            )
        )
        schema.help()

        help_calls = m_print_help.normalized_calls()
        help_calls = [{h["key"]: h["help"]} for h in help_calls]
        assert help_calls == [
            {"root": None},
            {"a": "Help for a"},
            {"b": "Help for b"},
            {"c": None},
            {"d": "Help for d"},
        ]

    def top_level_fields():
        def it_fetches_list_elem_type():
            schema = s(s.is_dict(elems=dict(a=s.is_list(s.is_int()))))
            tlf = schema.top_level_fields()
            assert tlf[0][0] == "a" and tlf[0][4] is int

        def it_fetches_user_data():
            schema = s(
                s.is_dict(
                    help="Help for a",
                    elems=dict(
                        b=s.is_int(help="Help for b", userdata="userdata_1"),
                        c=s.is_kws(d=s.is_int(help="Help for d")),
                    ),
                )
            )
            tlf = schema.top_level_fields()
            assert tlf[0][0] == "b" and tlf[0][3] == "userdata_1"

        zest()

    zest()
