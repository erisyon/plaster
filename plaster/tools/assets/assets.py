from plumbum import local
from plumbum.path.local import LocalPath
from plumbum import local, FG
from plaster.tools.schema import check
from plaster.tools.utils import utils


def get_user():
    user = local.env.get("RUN_USER")
    if user is None or user == "":
        raise Exception("User not found in $USER")
    return user


def validate_job_folder(job_folder):
    """
    job_folder can be Python symbols
    """
    basename = local.path(job_folder).name
    if not utils.is_symbol(basename):
        raise ValueError(
            "job name must be a lower-case Python symbol (ie start with [a-z_] followed by [a-z0-9_]"
        )

    return local.path(job_folder)
