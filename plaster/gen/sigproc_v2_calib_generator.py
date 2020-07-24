from munch import Munch
from plaster.gen import task_templates
from plaster.gen.base_generator import BaseGenerator
from plaster.tools.schema.schema import Schema as s
from plaster.tools.utils import utils


class SigprocV2CalibGenerator(BaseGenerator):
    schema = s(
        s.is_kws_r(
            **BaseGenerator.sigproc_source_schema.schema(),
            **BaseGenerator.label_set_schema.schema(),
            **BaseGenerator.scope_run_schema.schema(),
            **BaseGenerator.sigproc_v2_schema.schema(),
            **BaseGenerator.dye_names_schema.schema(),
        )
    )

    def generate(self):
        runs = []

        if len(self.sigproc_source) != 1:
            raise ValueError(f"Calibrations can have only one sigproc_source")
        sigproc_source = self.sigproc_source[0]

        ims_import_task = self.ims_imports(sigproc_source)
        sigproc_v2_task = task_templates.sigproc_v2(
            "dye_calib", self.calibration_file, self.instrument_subject_id
        )

        run = Munch(run_name=f"sigproc_v2_calib", **ims_import_task, **sigproc_v2_task)
        if self.force_run_name is not None:
            run.run_name = self.force_run_name

        self.report_section_run_object(run)
        template = "sigproc_v2_calib_template.ipynb"
        self.report_section_from_template(template)

        runs += [run]

        n_runs = len(runs)
        self.report_preamble(
            utils.smart_wrap(
                f"""
                # Sigproc V2 Calibration
                ## {n_runs} run(s) processed.
            """
            )
        )

        return runs
