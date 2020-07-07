#!/usr/bin/env python -u
"""
All commands that can be run in this project are available through this unified interface.
This should be run with the ./plaster.sh helper to get into the correct context.
"""
import os
import sys
import pandas as pd
from munch import Munch
from plumbum import FG, RETCODE, TEE, TF, cli, local
from plaster.tools.log.log import (
    colorful_exception,
    debug,
    error,
    important,
    info,
)
from plaster.tools.uniprot import uniprot
from plaster.tools.zap.zap import Context
from plaster.tools.utils.utils import (
    safe_list_get,
    save,
)
from plaster.tools.assets import assets


class CommandError(Exception):
    def __init__(self, retcode=None):
        self.retcode = retcode


class DoFuncs:
    def is_dev(self):
        return local.env.get("ERISYON_DEV") == "1"

    def folder_user(self):
        return local.env["FOLDER_USER"]

    def run_user(self):
        return local.env["RUN_USER"]

    def assert_env(self):
        pass
        # must_exist = ("ERISYON_ROOT", "ERISYON_HEADLESS")
        # found = 0
        # for e in must_exist:
        #     if e in local.env:
        #         found += 1
        #     else:
        #         error(f'Environment variable "{e}" not found.')
        #
        # if found != len(must_exist):
        #     raise CommandError

    def clear(self):
        local["clear"] & FG

    def _print_job_folders(self, file_list, show_plaster_json=True):
        """
        file_list is a list of munches [Munch(folder="folder", name="foo.txt", size=123, mtime=123456789)]
        """

        if len(file_list) == 0:
            print("No files found")
            return

        folders = {
            file.folder: Munch(folder=file.folder, size_gb=0, file_count=0,)
            for file in file_list
        }

        gb = 1024 ** 3
        total_gb = 0
        for file in file_list:
            folder = file.folder
            total_gb += file.size / gb
            folders[folder].size_gb += file.size / gb
            folders[folder].file_count += 1

        df = pd.DataFrame.from_dict(folders, orient="index")
        formatters = dict(
            size_gb="{:10.2f}".format,
            folder="{:<40.40s}".format,
            file_count="{:.0f}".format,
        )
        columns = ["folder", "size_gb", "file_count"]

        df = df.append(dict(folder="TOTAL", size_gb=total_gb), ignore_index=True)

        print(df.to_string(columns=columns, formatters=formatters))

    def print_local_job_folders(self):
        important("Local job folders:")

        root = local.path("./jobs_folder")
        self._print_job_folders(
            [
                Munch(
                    folder=(p - root)[0],
                    name=p.name,
                    size=int(p.stat().st_size),
                    mtime=int(p.stat().st_mtime),
                )
                for p in root.walk()
            ]
        )

    def validate_job_folder(self, job_folder, allow_run_folders=False):
        return assets.validate_job_folder(
            job_folder, allow_run_folders=allow_run_folders
        )

    def run_zests(self, **kwargs):
        coverage = kwargs.pop("coverage", False)
        important(f"Running zests{' (with coverage)' if coverage else ''}...")
        if coverage:
            raise NotImplementedError
            ret = local["coverage"]["run", "./gen_main.py", "zest"] & RETCODE(FG=True)
            if ret == 0:
                local["coverage"]["html"] & FG
                local["xdg-open"]("./.coverage_html/index.html")
        else:
            from zest.zest_runner import ZestRunner

            try:
                runner = ZestRunner(
                    include_dirs="./plaster/gen:./plaster/run:./plaster/tools", **kwargs
                )
                if runner.retcode != 0:
                    raise CommandError
                return 0
            except Exception as e:
                colorful_exception(e)
                return 1

    def run_nbstripout(self):
        """Strip all notebooks of output to save space in commits"""
        important("Stripping Notebooks...")
        result = (
            local["find"][
                ".",
                "-type",
                "f",
                "-not",
                "-path",
                "*/\.*",
                "-name",
                "*.ipynb",
                "-print",
            ]
            | local["xargs"]["nbstripout"]
        ) & TF(FG=True)

        if not result:
            raise CommandError

    def run_docker_build(self, docker_tag, quiet=False):
        important(f"Building docker tag {docker_tag}")
        with local.env(LANG="en_US.UTF-8"):
            args = [
                "build",
                "-t",
                f"erisyon:{docker_tag}",
                "-f",
                "./scripts/main_env.docker",
            ]
            if quiet:
                args += ["--quiet"]
            args += "."
            local["docker"][args] & FG


class DoCommand(cli.Application, DoFuncs):
    def main(self):
        return


@DoCommand.subcommand("run_notebook")
class RunNotebookCommand(cli.Application, DoFuncs):
    def main(self, notebook_path):
        self.assert_env()
        local["jupyter"][
            "nbconvert",
            "--to",
            "html",
            "--execute",
            notebook_path,
            "--ExecutePreprocessor.timeout=1800",
        ] & FG


@DoCommand.subcommand("test")
class TestCommand(cli.Application, DoFuncs):
    no_clear = cli.Flag("--no_clear", help="Do not clear screen")
    verbose = cli.SwitchAttr("--verbose", int, default=1, help="Verbosity 0,1,2")
    coverage = cli.Flag("--coverage", help="Run coverage and show in local HTML")
    debug_mode = cli.Flag("--debug_mode", help="Put zap into debug_mode")
    run_groups = cli.SwitchAttr(
        "--run_groups", str, help="Comma separated list of groups"
    )
    disable_shuffle = cli.Flag(
        "--disable_shuffle", help="If set, do not shuffle test order"
    )

    def main(self, *args):
        self.assert_env()
        if not self.no_clear:
            self.clear()

        match_string = safe_list_get(args, 0)

        with Context(debug_mode=self.debug_mode):
            return self.run_zests(
                verbose=self.verbose,
                match_string=match_string,
                coverage=self.coverage,
                run_groups=self.run_groups,
            )


@DoCommand.subcommand("jupyter")
class JupyterCommand(cli.Application, DoFuncs):
    ip = cli.SwitchAttr("--ip", str, default="0.0.0.0", help="ip to bind to")
    port = cli.SwitchAttr("--port", int, default="8080", help="port to bind to")

    def main(self, *args):
        self.assert_env()
        os.execlp(
            "jupyter",
            "jupyter",
            "notebook",
            f"--ip={self.ip}",
            f"--port={self.port}",
            "--allow-root",
            *args,
        )


if __name__ == "__main__":
    try:
        DoCommand.subcommand("gen", "plaster.gen.gen_main.GenApp")
        DoCommand.subcommand("run", "plaster.run.run_main.RunApp")
        DoCommand.run()
    except (KeyboardInterrupt):
        print()  # Add an extra line because various thing terminate with \r
        sys.exit(1)
    except Exception as e:
        colorful_exception(e)
        sys.exit(1)
