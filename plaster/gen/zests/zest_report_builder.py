import json

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from plaster.gen import report_builder
from zest import zest


def execute_notebook(nb, timeout=5):
    """
    Runs a notebook, passed as a dict (in the format returned by ReportBuilder.report_assemble).

    Timeout is set to 5 seconds, if any test needs to increase this value then these tests probably shouldn't be run as part of the normal unit tests.
    """
    loaded_nb = nbformat.reads(json.dumps(nb), as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    return ep.preprocess(loaded_nb)


def zest_report_builder():
    """
    TODO: add test for template section
    """
    builder = None

    def _execute():
        # Assembles the report for the builder and executes it
        nb = builder.report_assemble()
        return execute_notebook(nb)

    def _before():
        # Generates a new builder before each subtest
        nonlocal builder
        builder = report_builder.ReportBuilder()

    def it_generates_code_section():
        code = "print(0)"
        builder.add_report_section(report_builder.SectionType.CODE, [code])
        nb = _execute()
        cells = nb[0]["cells"]
        for cell in cells:
            if cell["cell_type"] == "code":
                if code in cell["source"]:
                    assert any("0\n" in output["text"] for output in cell["outputs"])
                    break

    def it_generates_markdown_section():
        markdown = "# Hello"
        builder.report_section_markdown(markdown)
        nb = _execute()
        cells = nb[0]["cells"]
        assert any(
            cell["cell_type"] == "markdown" and markdown in cell["source"]
            for cell in cells
        )

    def it_generates_preamble_section():
        markdown = "# Hello"
        builder._report_preamble = markdown
        nb = _execute()
        cells = nb[0]["cells"]
        assert any(
            cell["cell_type"] == "markdown" and markdown in cell["source"]
            for cell in cells
        )

    zest()
