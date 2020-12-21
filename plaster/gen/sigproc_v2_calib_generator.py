from munch import Munch
from plaster.gen import task_templates
from plaster.gen.base_generator import BaseGenerator
from plaster.tools.schema.schema import Schema as s
from plaster.run.sigproc_v2 import sigproc_v2_common
from plaster.tools.utils import utils


class SigprocV2CalibGenerator(BaseGenerator):
    """
    PSF Calib takes a z-stack movie of single dye-count
    (future: multi-channel single dye count).
    """

    schema = s(s.is_kws_r(**BaseGenerator.sigproc_v2_calib_schema.schema(),))

    def generate(self):
        runs = []

        assert isinstance(self.sigproc_source, str)

        ims_import_task = task_templates.ims_import(
            self.sigproc_source, is_movie=(self.mode == "psf")
        )

        # See note above. Only one option at moment
        modes = dict(illum=sigproc_v2_common.SIGPROC_V2_ILLUM_CALIB,)
        mode = modes.get(self.mode)
        assert mode is not None

        sigproc_v2_calib_task = task_templates.sigproc_v2_calib(
            mode=mode, instrument_identity=self.instrument_identity
        )

        run = Munch(
            run_name=f"sigproc_v2_calib", **ims_import_task, **sigproc_v2_calib_task,
        )

        if self.force_run_name is not None:
            run.run_name = self.force_run_name

        self.report_section_run_object(run)
        template = "sigproc_v2_instrument_calib_template.ipynb"
        self.report_section_from_template(template)

        runs += [run]

        self.report_preamble(
            utils.smart_wrap(
                f"""
                # Sigproc V2 Instrument Calibration
                """,
                width=None,
            )
        )

        return runs
