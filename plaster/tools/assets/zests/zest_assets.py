import tempfile
import plumbum
from plumbum import local
from zest import zest
from plaster.tools.assets import assets
from plaster.tools.schema import check


def zest_validate_job_folder():
    job_name = "__test_job1"
    job1 = local.path("./jobs_folder") / job_name
    run1 = local.path("./jobs_folder") / job_name / "run1"

    def it_raises_if_not_symbol():
        with zest.raises(ValueError):
            assets.validate_job_folder("foo bar")

    def it_returns_a_plumbum_path():
        assert isinstance(
            assets.validate_job_folder("foo_bar"), plumbum.path.local.LocalPath
        )

    zest()
