import sys

from plaster.tools.log.log import (
    colorful_exception,
    debug,
    error,
    important,
    info,
    prof,
)
from plumbum import BG, FG, TEE, TF, cli, local


class PlasCommand(cli.Application):
    PROGNAME = "plas"


if __name__ == "__main__":
    try:
        with local.cwd(local.env["PLASTER_ROOT"]):
            PlasCommand.subcommand("gen", "plaster.gen.gen_main.GenApp")
            PlasCommand.subcommand("run", "plaster.run.run_main.RunApp")
            PlasCommand.run()
    except KeyboardInterrupt:
        print()  # Add an extra line because various thing terminate with \r
        sys.exit(1)
    except Exception as e:
        colorful_exception(e)
        sys.exit(1)
