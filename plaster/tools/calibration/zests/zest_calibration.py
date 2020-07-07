import numpy as np
from munch import Munch
from zest import zest
from plaster.tools.calibration import calibration as calib
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.schema import schema


def zest_calibration():
    def validation():
        def it_checks_prop():
            with zest.raises(TypeError):
                Calibration({"not_a_property.instrument.sn1234": 1})

        def it_checks_subject_type():
            with zest.raises(TypeError):
                Calibration({"p_failure_to_bind_amino_acid.not_a_subject.sn1234": 1})

        def it_checks_subject_id():
            with zest.raises(TypeError):
                Calibration({"p_failure_to_bind_amino_acid.label[C].1234": 1})

        def it_checks_propsub():
            with zest.raises(TypeError):
                Calibration({"p_failure_to_bind_amino_acid.instrument.sn1234": 1})

        def it_checks_value():
            with zest.raises(TypeError):
                Calibration(
                    {"p_failure_to_bind_amino_acid.label[C].sn1234": "not_a_float"}
                )

        def it_checks_variable_subtype():
            with zest.raises(TypeError):
                Calibration({"p_failure_to_bind_amino_acid.label[foo].sn1234": 1.0})

        def it_checks_metadata():
            with zest.raises(TypeError):
                Calibration({"metadata.instrument.sn1234": "not a dict"})

        zest()

    def it_allows_metadata():
        c = Calibration({"metadata.instrument.sn1234": dict(a=1)})
        assert c["metadata.instrument.sn1234"]["a"] == 1

    def it_filters():
        c = Calibration(
            {
                "p_failure_to_bind_amino_acid.label[C].sn1234": 0.5,
                "p_failure_to_bind_amino_acid.label[C].sn0": 1.0,
            }
        )
        c.filter_subject_ids({"sn1234"})
        assert len(c) == 1
        assert c["p_failure_to_bind_amino_acid.label[C]"] == 0.5

    def it_adds():
        c = Calibration({"p_failure_to_bind_amino_acid.label[C].batch_2020_03_01": 1.0})
        c.add({"p_failure_to_attach_to_dye.label[C].batch_2020_03_01": 2.0})
        assert len(c) == 2

    def it_sets_subject_id():
        c = Calibration(
            {
                "p_failure_to_bind_amino_acid.label[C]": 1.0,
                "p_failure_to_attach_to_dye.label[C]": 2.0,
            }
        )
        c.set_subject_id("test")
        assert c["p_failure_to_bind_amino_acid.label[C].test"] == 1.0
        assert c["p_failure_to_attach_to_dye.label[C].test"] == 2.0

    zest()
