from munch import Munch
from plaster.tools.schema.schema import Schema as s
from plaster.tools.utils import utils
from plumbum import local
from plaster.gen import task_templates
from plaster.gen.errors import ValidationError
from plaster.gen.base_generator import BaseGenerator
from plaster.gen.sigproc_v1_generator import SigprocV1Generator

modes = ("gain", "vpe")


class CalibNNGenerator(SigprocV1Generator):
    """
    Import calibration runs
    """

    schema = s(
        s.is_kws_r(
            **SigprocV1Generator.schema.schema(),
            **BaseGenerator.scope_run_schema.schema(),
            mode=s.is_str(
                help=f"Current modes are: [{', '.join(modes)}]", userdata=dict(cli=True)
            ),
            channel=s.is_list(s.is_int(), help=f"Channel list to include"),
            dye_names=s.is_str(
                help="Dye names of each channel; will be saved with this scope.",
                userdata=dict(cli=True),
            ),
            scope_name=s.is_str(
                help="Scope name, will be saved with this scope.",
                userdata=dict(cli=True),
            ),
        )
    )

    def generate(self):
        runs = []
        sigproc_tasks = self.sigprocs_v1()

        if len(self.sigproc_source) != 1:
            raise ValueError(f"Calibrations can have only one sigproc_source")

        if self.mode not in modes:
            raise ValueError(f"Unknown calib mode {self.mode}")

        sigproc_task = sigproc_tasks[0]
        calib_task = task_templates.calib_nn(
            sigproc_relative_path=f"../sigproc_v1",
            mode=self.mode,
            n_pres=self.n_pres,
            n_mocks=self.n_mocks,
            n_edmans=self.n_edmans,
            dye_names=self.dye_names,
            scope_name=self.scope_name,
            channels=self.channel,
        )

        run = Munch(run_name=f"calib_{self.mode}", **sigproc_task, **calib_task,)

        self.report_section_run_object(run, to_load=["sim", "calib"])
        calib_template = "calib_nn_template.ipynb"
        self.report_section_from_template(calib_template)

        runs += [run]

        n_runs = len(runs)
        self.report_preamble(
            utils.smart_wrap(
                f"""
                # Calib Overview
                ## {n_runs} run(s) processed.
            """
            )
        )

        return runs
