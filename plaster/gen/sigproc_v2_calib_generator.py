from munch import Munch
from plaster.gen import task_templates
from plaster.gen.base_generator import BaseGenerator
from plaster.tools.schema.schema import Schema as s
from plaster.run.sigproc_v2 import sigproc_v2_common


class SigprocV2CalibGenerator(BaseGenerator):
    """
    PSF Calib takes a z-stack movie of single dye-count
    (future: multi-channel single dye count).
    """

    schema = s(
        s.is_kws_r(
            **BaseGenerator.sigproc_v2_schema.schema(),
            mode=s.is_str(options=["psf", "illum"]),
        )
    )
    has_report = False

    def generate(self):
        runs = []

        if len(self.sigproc_source) != 1:
            raise ValueError(f"Calibrations can have only one sigproc_source")

        ims_import_task = task_templates.ims_import(
            self.sigproc_source[0], is_movie=(self.mode == "psf")
        )

        modes = dict(
            psf=sigproc_v2_common.SIGPROC_V2_PSF_CALIB,
            illum=sigproc_v2_common.SIGPROC_V2_ILLUM_CALIB,
        )
        mode = modes.get(self.mode)
        assert mode is not None

        sigproc_v2_calib_task = task_templates.sigproc_v2_calib(
            self.calibration_file, mode=mode,
        )

        run = Munch(
            run_name=f"sigproc_v2_calib", **ims_import_task, **sigproc_v2_calib_task,
        )

        if self.force_run_name is not None:
            run.run_name = self.force_run_name

        # self.report_section_run_object(run)
        # template = "sigproc_v2_instrument_calib_template.ipynb"
        # self.report_section_from_template(template)

        runs += [run]

        n_runs = len(runs)
        # self.report_preamble(
        #     utils.smart_wrap(
        #         f"""
        #         # Sigproc V2 Instrument Calibration
        #         ## {n_runs} run(s) processed.
        #     """
        #     )
        # )

        return runs
