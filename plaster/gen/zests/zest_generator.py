from munch import Munch
from plaster.gen.base_generator import BaseGenerator
from plaster.gen.helpers import task_rename
from plaster.tools.log.log import debug
from plaster.tools.schema.schema import Schema as s
from zest import zest


def zest_BaseGenerator():
    gen_klass = None

    def _before():
        class TestGen(BaseGenerator):
            schema = s(
                s.is_kws_r(
                    n_edmans=s.is_int(help="Number of Edman cycles"),
                    n_pres=s.is_int(),
                    protease=s.is_list(s.is_str()),
                    label_set=s.is_list(s.is_str()),
                )
            )

            defaults = Munch(n_pres=0)

        nonlocal gen_klass
        gen_klass = TestGen

    def it_accepts_required_fields():
        g = gen_klass(n_edmans=1, n_pres=2, protease=[], label_set=[])
        assert g.n_edmans == 1 and g.n_pres == 2

    def it_applies_defaults():
        g = gen_klass(n_edmans=1, protease=[], label_set=[])  # Missing n_pres
        assert g.n_edmans == 1 and g.n_pres == 0

    def zest_label_permutate():
        class Gen(BaseGenerator):
            schema = s(s.is_kws(label_set=s.is_list(s.is_str())))

        def zest_label_str_permutate():
            gen = Gen()

            def it_permutates_label_strs():
                perms = gen._label_str_permutate("A,B,CD: 2")
                assert perms == [("A", "B"), ("A", "CD"), ("B", "CD")]

            def it_permutates_and_adds_label_strs():
                perms = gen._label_str_permutate("A,B,CD: 2 + S")
                assert perms == [("A", "B", "S"), ("A", "CD", "S"), ("B", "CD", "S")]
                perms = gen._label_str_permutate("A,B,CD: 2 + S,T")
                assert perms == [
                    ("A", "B", "S", "T"),
                    ("A", "CD", "S", "T"),
                    ("B", "CD", "S", "T"),
                ]
                perms = gen._label_str_permutate("A,B,CD: 2 + S[p]T[p]")
                assert perms == [
                    ("A", "B", "S[p]T[p]"),
                    ("A", "CD", "S[p]T[p]"),
                    ("B", "CD", "S[p]T[p]"),
                ]

            def it_permutates_default():
                perms = gen._label_str_permutate("A,B,CD")
                assert perms == [("A", "B", "CD")]

            def it_removes_whitespace():
                perms = gen._label_str_permutate("A , B  , CD")
                assert perms == [("A", "B", "CD")]

            def it_raises_on_too_many_colons():
                with zest.raises(ValueError):
                    gen._label_str_permutate("A,B,CD: 3: 4")

            def it_raises_on_out_of_range_count():
                with zest.raises(ValueError):
                    gen._label_str_permutate("A,B,CD: 4")
                with zest.raises(ValueError):
                    gen._label_str_permutate("A,B,CD: -1")

            zest()

        def zest_label_set_permutate():
            def it_defaults_arguments_to_self():
                gen = Gen(label_set=["AB,C"])
                res = gen.label_set_permutate()
                assert res == [("AB", "C")]

            def it_flattens():
                gen = Gen(label_set=["A,B,C:2", "AB"])
                res = gen.label_set_permutate()
                assert res == [("A", "B"), ("A", "C"), ("B", "C"), ("AB",)]

            zest()

        zest()

    def it_permutes_labels_and_proteases():
        def it_defaults_arguments_to_self():
            gen = gen_klass(
                n_edmans=1, label_set=["A,B,C:2", "C"], protease=["p0", "p1"], scheme=[]
            )
            perms = list(gen.run_parameter_permutator())
            # assert perms == [
            #     ("p0", ("A", "B")),
            #     ("p0", ("A", "C")),
            #     ("p0", ("B", "C")),
            #     ("p0", ("C",)),
            #     ("p1", ("A", "B")),
            #     ("p1", ("A", "C")),
            #     ("p1", ("B", "C")),
            #     ("p1", ("C",)),
            # ]

        # def it_accepts_arguments():
        #     gen = gen_klass(n_edmans=1, protease=[], label_set=[])
        #     perms = list(
        #         gen.protease_labels_permutator(
        #             label_set=["A", "B"], protease=["p0", "p1"]
        #         )
        #     )
        #     assert perms == [("p0", ("A",)), ("p0", ("B",)), ("p1", ("A",)), ("p1", ("B",))]
        #
        # def it_handles_protease_none():
        #     gen = gen_klass(n_edmans=1, protease=[], label_set=[])
        #     perms = list(
        #         gen.protease_labels_permutator(label_set=["A", "B"], protease=None)
        #     )
        #     assert perms == [(None, ("A",)), (None, ("B",))]

        zest()

    def it_permutes_schemes():
        def it_parses_correct_arguments():
            gen = gen_klass(
                n_edmans=1,
                label_set=[],
                protease=[],
                scheme=["trypsin/C,M,Y", "trypsin/DE,Y,C,K:2+S"],
            )
            perms = list(gen.run_parameter_permutator())

        def it_raises_on_incorrect_arguments():
            bad_schemes = ["/C,M,Y", "a/b/c", "", "protease/"]
            for bad_scheme in bad_schemes:
                gen = gen_klass(
                    n_edmans=1, label_set=[], protease=[], scheme=[bad_scheme],
                )
                with zest.raises():
                    perms = list(gen.run_parameter_permutator())

        zest()

    def it_renames_task_blocks():
        def it_raises_on_bad_task():
            with zest.raises(AssertionError):
                task_rename(
                    Munch(a=Munch(), b=Munch()), "foo"
                )  # Munch doesn't have a unitary root key

        def it_renames_root_key():
            task = Munch(a=Munch(b=1))
            task_rename(task, "foo")
            assert task.foo.b == 1 and not task.get("a")

        zest()

    zest()
