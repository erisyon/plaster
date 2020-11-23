from munch import Munch
from plaster.gen import task_templates
from plaster.tools.log import log
from zest import zest


def zest_task_templates_prep():
    def _fake_seq(abundance):
        return Munch(id="a", seqstr="a", abundance=abundance)

    def it_normalizes_abundance_data():
        with zest.mock(log.info) as m_log:
            ret = task_templates.prep([_fake_seq(5), _fake_seq(10)], None, None)

        assert [p.abundance for p in ret.prep.parameters.proteins] == [1.0, 2.0]

    zest()
