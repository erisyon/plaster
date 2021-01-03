#!/usr/bin/env python

"""
The main entrypoint for plaster.

Note that this is typically run through the ./p helper instead of directly.
"""

import arrow
from plumbum import cli, local
from plaster.tools.utils import utils
from plaster.tools.assets import assets
from plaster.tools.zap import zap
from plaster.tools.log.log import (
    error,
    important,
    debug,
    metrics,
)
from plaster.run.run import RunExecutor


class RunApp(cli.Application):
    """
    Run plaster components as directed by a plaster_run.yaml file in the given path.

    If the specified src_dir is not in ./jobs_folder then you will be
    asked if you want to create a symbolic link within jobs_folder
    to the specified src_dir.
    """

    limit = cli.SwitchAttr(
        ["--limit"],
        str,
        help="Comma separated list of targets to build (skipping upstream requirements)",
    )
    force = cli.Flag(
        "--force",
        default=False,
        help="If True, targets will be rebuilt unconditionally",
    )
    clean = cli.Flag(
        "--clean", default=False, help="If True, the targets will cleaned but not built"
    )
    cpu_limit = cli.SwitchAttr(
        "--cpu_limit", int, default=-1, help="Set to limit to n_cpus"
    )
    skip_reports = cli.Flag(
        "--skip_reports", default=False, help="Do not run the report notebook(s)"
    )
    debug_mode = cli.Flag("--debug_mode", default=False, help="Easier debug tracing")
    skip_s3 = cli.Flag(
        "--skip_s3", default=False, help="Skip S3 translation (assume already cached)"
    )
    continue_on_error = cli.Flag("--continue_on_error", default=False)

    n_fields_limit = cli.SwitchAttr(
        "--n_fields_limit", int, help="Set to limit on number of fields to process"
    )

    def run_ipynb(self, ipynb_path):
        # Note: the timeout has been set to 8 hours to facilitate reports for
        # huge jobs (e.g. 100+ runs).
        important(f"Executing report notebook {ipynb_path}")
        local["jupyter"](
            "nbconvert",
            "--to",
            "html",
            "--execute",
            ipynb_path,
            "--ExecutePreprocessor.timeout=28800",
        )

    def main(self, job_folder=None):
        switches = utils.plumbum_switches(self)

        if job_folder is None:
            error(f"No job_folder was specified")
            return 1

        important(
            f"Plaster run {job_folder} limit={self.limit} started at {arrow.utcnow().format()}"
        )

        job_folder = assets.validate_job_folder(job_folder)
        if not job_folder.exists():
            error(f"Unable to find the path {job_folder}")
            return 1

        # Find all the plaster_run.yaml files. They might be in run subfolders
        found = list(job_folder.walk(filter=lambda p: p.name == "plaster_run.yaml"))
        run_dirs = [p.dirname for p in found]

        if len(run_dirs) == 0:
            error(
                "Plaster: Nothing to do because no run_dirs have plaster_run.yaml files"
            )
            return 1

        # A normal run where all happens in this process
        failure_count = 0
        for run_dir_i, run_dir in enumerate(sorted(run_dirs)):

            metrics(
                _type="plaster_start",
                run_dir=run_dir,
                run_dir_i=run_dir_i,
                run_dir_n=len(run_dirs),
                **switches,
            )
            important(
                f"Starting run subdirectory {run_dir}. {run_dir_i + 1} of {len(run_dirs)}"
            )

            try:
                with zap.Context(cpu_limit=self.cpu_limit, debug_mode=self.debug_mode):
                    run = RunExecutor(run_dir).load()
                    if "_erisyon" in run.config:
                        metrics(_type="erisyon_block", **run.config._erisyon)

                    failure_count += run.execute(
                        force=self.force,
                        limit=self.limit.split(",") if self.limit else None,
                        clean=self.clean,
                        n_fields_limit=self.n_fields_limit,
                        skip_s3=self.skip_s3,
                    )
            except Exception as e:
                failure_count += 1
                if not self.continue_on_error:
                    raise e

        if (
            failure_count == 0
            and self.limit is None
            and not self.clean
            and not self.skip_reports
        ):
            # RUN reports
            report_src_path = job_folder / "report.ipynb"
            report_dst_path = job_folder / "report.html"
            if (
                self.force
                or report_src_path.exists()
                and utils.out_of_date(report_src_path, report_dst_path)
            ):
                self.run_ipynb(report_src_path)
            return 0

        return failure_count


if __name__ == "__main__":
    # This is ONLY executed if you do not "main.py" because
    # main.py imports this file as a subcommand
    RunApp.run()
