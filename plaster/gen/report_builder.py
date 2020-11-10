from enum import Enum

from munch import Munch
from plaster.tools.utils import utils
from plumbum import local


class SectionType(str, Enum):
    """
    By inheriting from str + Enum, we get an Enum that will json.dumps to a string

    TODO: there appears to be "section" types and "cell" types. Do we need another enum for "cell" types? Is the set of valid options different?
    """

    MARKDOWN = "markdown"
    TEMPLATE = "template"
    CODE = "code"


class ReportBuilder:
    report_metadata = Munch(
        metadata=Munch(
            kernelspec=Munch(
                display_name="Python 3", language="python", name="python3"
            ),
            language_info=Munch(
                codemirror_mode=Munch(name="ipython", version=3),
                file_extension=".py",
                mimetype="text/x-python",
                name="python",
                nbconvert_exporter="python",
                pygments_lexer="ipython3",
                version="3.6.7",
            ),
        ),
        nbformat=4,
        nbformat_minor=2,
    )

    code_block = Munch(
        cell_type=SectionType.CODE,
        execution_count=None,
        metadata=Munch(),
        outputs=[],
        source=[],
    )

    markdown_block = Munch(cell_type=SectionType.MARKDOWN, metadata=Munch(), source=[])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._report_sections = []
        self._report_preamble = ""

    def add_report_section(self, section_type: SectionType, content):
        self._report_sections += [(section_type, content)]

    def report_preamble(self, markdown):
        """A a preamble in markdown format"""
        self._report_preamble = markdown

    def report_section_markdown(self, markdown):
        self.add_report_section(SectionType.MARKDOWN, markdown)

    def report_section_run_object(self, run):
        self.add_report_section(
            SectionType.CODE, [f'run = RunResult("./{run.run_name}")']
        )

    def report_section_first_run_object(self):
        self.add_report_section(SectionType.CODE, [f"run = job.runs[0]"])

    def report_section_job_object(self):
        self.add_report_section(SectionType.CODE, [f'job = JobResult(".")'])

    def _markdown_to_markdown_block(self, markdown):
        lines = [f"{line}\n" for line in markdown.split("\n")]
        block = Munch(**self.markdown_block)
        block.source = lines
        return block

    def report_section_run_array(self, runs, to_load=None):
        to_load_string = "" if to_load is None else f", to_load={to_load}"
        run_names = [run.run_name for run in runs]
        self.add_report_section(
            SectionType.CODE,
            [
                f"run_names = {run_names}\n"
                f'runs = [RunLoader(f"./{{name}}"{to_load_string}) for name in run_names]'
            ],
        )

    def _nb_template_path(self, template_name):
        return local.path(__file__) / "../nb_templates" / template_name

    def report_section_from_template(self, template_name):
        """Write the report from its pieces"""
        self._report_sections += [(SectionType.TEMPLATE, template_name)]

    def report_assemble(self):
        """Assemble the report from its pieces. A giant Munch is returned"""
        report = Munch(**self.report_metadata)
        report.cells = []

        preamble_block = self._markdown_to_markdown_block(self._report_preamble)
        report.cells += [preamble_block]

        # LOAD all templates
        templates_by_name = {}
        for section_type, section_data in self._report_sections:
            if section_type == SectionType.TEMPLATE:
                templates_by_name[section_data] = utils.json_load_munch(
                    self._nb_template_path(section_data)
                )

        # FIND all of the @IMPORT-MERGE blocks
        import_merge = []
        for _, template in templates_by_name.items():
            for cell in template.cells:
                if cell.cell_type == SectionType.CODE:
                    first_line = utils.safe_list_get(cell.source, 0, "")
                    if "# @IMPORT-MERGE" in first_line:
                        for line in cell.source:
                            if "import" in line:
                                import_merge += [line]

        import_merge += ["from plaster.tools.zplots import zplots\n"]
        import_merge = sorted(list(set(import_merge))) + ["z = zplots.setup()"]
        import_block = Munch(**self.code_block)
        import_block.source = import_merge
        report.cells += [import_block]

        for section_type, section_data in self._report_sections:
            if section_type == SectionType.CODE:
                lines = section_data
                block = Munch(**self.code_block)
                block.source = lines
                report.cells += [block]

            elif section_type == SectionType.MARKDOWN:
                block = self._markdown_to_markdown_block(section_data)
                report.cells += [block]

            elif section_type == SectionType.TEMPLATE:
                file_path = section_data
                template = templates_by_name[file_path]
                for cell in template.cells:
                    if cell.cell_type == SectionType.CODE:
                        first_line = utils.safe_list_get(cell.source, 0, "")

                        if (
                            "@IMPORT-MERGE" not in first_line
                            and "@REMOVE-FROM-TEMPLATE" not in first_line
                        ):
                            block = Munch(**self.code_block)
                            block.source = cell.source
                            report.cells += [block]

                    if cell.cell_type == SectionType.MARKDOWN:
                        block = Munch(**self.markdown_block)
                        block.source = cell.source
                        report.cells += [block]

        return report
