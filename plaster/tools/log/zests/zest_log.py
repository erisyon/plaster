import time
import re
import os
import numpy as np
import inspect
from plaster.tools.log import log
from zest import zest


def where():
    w = inspect.getframeinfo(inspect.stack()[1][0])
    return w.filename, w.lineno, os.path.basename(w.filename)


def zest_make_prefix():
    def it_encodes_the_level():
        assert log._make_prefix(9, -1)[0] == "9"

    def it_references_the_caller_and_lineno():
        _, lineno, basename = where()
        prefix = log._make_prefix(1, -2)
        assert f"{basename}:{lineno + 1}" in prefix

    def it_adds_a_fixed_width_timestamp():
        with zest.mock(log._timestamp, returns=1.123_456_789_012_34):
            prefix = log._make_prefix(1, -1)
            assert "         1.123457 " in prefix

    def it_adds_a_pid():
        prefix = log._make_prefix(1, -1)
        assert f" {os.getpid()} " in prefix

    zest()


def zest_exception():
    filename, lineno = None, None

    def _generate_exception(to_raise):
        nonlocal filename, lineno
        try:
            filename, lineno, _ = where()
            raise to_raise
        except Exception as e:
            return log._make_exception(e)

    def it_includes_and_increments_a_reference_counter_in_the_first_line():
        log.exception_count = 0
        lines1 = _generate_exception(Exception())
        lines2 = _generate_exception(Exception())
        assert "[1]" in lines1[0]
        assert "[2]" in lines2[0]

    def it_includes_the_exception_class():
        lines = _generate_exception(ValueError())
        assert "ValueError" in lines[0]

    def it_includes_the_exception_content():
        c = "this is some content"
        lines = _generate_exception(ValueError(c))
        assert c in lines[0]

    def it_includes_a_reference_to_the_caller_in_the_stack_trace():
        lines = _generate_exception(Exception())
        find_line = f"line {lineno + 1}"
        assert any([(filename in line and find_line in line) for line in lines])

    def it_wraps_start_and_end_lines():
        lines = _generate_exception(Exception())
        assert "Start" in lines[0] and "End" in lines[-1]

    zest()


def zest_input_request():
    zest.stack_mock(log._interactive_emit_line)

    def it_calls_input_when_not_headless():
        with zest.mock(log.is_headless, returns=False):
            with zest.mock(log._input, returns="ret test 1") as m_input:
                ret = log.input_request("test1", "when headless")
                assert m_input.called_once()
                assert ret == "ret test 1"

    def it_handles_headless_mode():
        m_input = zest.stack_mock(log._input)
        zest.stack_mock(log.is_headless, returns=True)

        def it_does_not_call_input_when_headless():
            ret = log.input_request("test2", "when headless")
            assert m_input.not_called()
            assert ret == "when headless"

        def it_raises_when_headless_and_theres_an_exception_passed():
            ret = log.input_request("test2", "when headless")
            assert m_input.not_called()
            assert ret == "when headless"

        zest()

    zest()
