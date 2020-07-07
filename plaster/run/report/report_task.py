from plumbum import local
from plumbum import path as path_utils
from plaster.tools.pipeline.pipeline import PipelineTask


class ReportTask(PipelineTask):
    def start(self):
        template = self.config.parameters.get("template")
        if template is not None:
            notebook_templates = (
                local.path(local.env["ERISYON_ROOT"]) / "projects/plaster/notebooks"
            )
            path_utils.copy(notebook_templates / template, ".")
