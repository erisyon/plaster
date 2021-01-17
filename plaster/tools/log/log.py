#!/usr/bin/env python

"""
Log wrapper

Features:
    * Simplified states over logger / glog:
        info
        error
        exception
    * info mode is special because it does not prefix in non-headless (interactive) mode. This is useful
      so that you can use it for things like "progress indicators" that, when run in interactive mode,
      you don't want the user to see any ugly prefixes.
    * No flags, all functionality determined by environment variables
        * ERISYON_HEADLESS=1
    * Handy debug() that looks up the variable names from the caller like:
        a = 1; b = 2
        debug(a, b)
        # Prints: somefile.py:14] a=1, b=2
    * Includes input requests that does something smart when headless.

Tasks:
    * When run as a standalone bin it wraps the call and allows for all non-standard outputs of the
      child process to get headers. This is useful when there are annoying traces coming from libraries
    * Tag scoping. Handy for when you need to associate all message with a tag, for example, when
      processing a certain message in a worker you would like to know which message caused a problem
    * Multi-line messages get sub-labels with clear boundaries
"""
import fcntl
import inspect
import json
import os
import re
import struct
import sys
import termios
import threading
import time
import traceback

import arrow
import numpy as np
import pandas as pd

INFO = 1
ERROR = 2
EXCEPTION = 3


# global number of exception_count useful for trace unwrangling when logs get merged
exception_count = 0
no_colors = False
blue = "\u001b[34m"
yellow = "\u001b[33m"
red = "\u001b[31m"
green = "\u001b[32m"
gray = "\u001b[30;1m"
cyan = "\u001b[36m"
magenta = "\u001b[35m"
bold = "\u001b[1m"
reset = "\u001b[0m"


# Not really sure when this should be called because we want to use this
# in a threadpoolexectuor context. Seems odd to allocate the local here in
# the module. Why isn't there a Python singleton for this?
thread_local = threading.local()


def is_headless():
    """Mock-point"""
    return os.environ.get("ERISYON_HEADLESS", "0") == "1"


def _input():
    """Mock-point"""
    return input()


def _arg_type_str(arg):
    s = str(type(arg))
    if s.startswith("<class '"):
        return s[8:-2]
    return s


def _emit_line(line, strip=True, newline=True):
    """Mock-point"""
    if strip:
        line = line.strip()
    sys.stderr.write(line)
    if newline:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _timestamp():
    """Mock-point"""
    t = arrow.utcnow()
    return t.timestamp + t.microsecond / 1e6


def _make_prefix(level, call_depth):
    """
    Generate prefix for messages (context, etc)
        level: int
            Trace level, see constants
        depth: int
            A negative offset saying how far back in the stack to pull the "true" source line
    """

    caller = traceback.extract_stack()[call_depth]
    location = f"{os.path.basename(caller.filename)}:{caller.lineno}"

    tags = ""  # TASK

    return f"{level} {_timestamp():17.6f} {os.getpid()} {location}{tags}] "


def _make_exception(e, msg=""):
    """
    Format exceptions in a useful way with a stack trace
    Returns:
        A list of lines
    """
    global exception_count
    exception_count += 1

    lines = []

    def head_foot(which):
        return (
            f">>>>>>>>>>>>>>>>>>>>> {which} [{exception_count}] "
            f"{e.__class__.__name__}({e}) >>>>>>>>>>>>>>>>>>>>>"
        )

    lines += [head_foot("Start"), msg]

    formatted = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)

    # Oddly, the formatted has sub-lines (more than one \n per row)
    for line in formatted:
        lines += [sub_line for sub_line in line.strip().split("\n")]

    lines += [head_foot("End")]

    return lines


def _interactive_emit_line(level, color, msg, **kwargs):
    """Only add prefix if in headless mode"""
    prefix = ""
    if is_headless():
        prefix = _make_prefix(level, -4)

    if no_colors or color is None:
        _emit_line(prefix + msg, **kwargs)
    else:
        _emit_line(prefix + color + msg + reset, **kwargs)


def info(msg, **kwargs):
    """Use this even in interactive mode because it doesn't add prefix unless headless."""
    _interactive_emit_line(INFO, None, msg, **kwargs)


def important(msg):
    """Use this even in interactive mode because it doesn't add prefix unless headless."""
    _interactive_emit_line(INFO, yellow, msg)


def success(msg):
    """Use this even in interactive mode because it doesn't add prefix unless headless."""
    _interactive_emit_line(INFO, green, msg)


def error(msg):
    """Use this even in interactive mode because it doesn't add prefix unless headless."""
    _interactive_emit_line(ERROR, red, msg)


def metrics(**kws):
    """In headless mode, this will emit the dict of vals as json in a way that can be easily extracted"""
    if is_headless():
        prefix = _make_prefix(INFO, -3)
        _emit_line(prefix + "@~@" + json.dumps(kws, skipkeys=True))


def exception(e, msg=""):
    """See _make_exception"""
    assert isinstance(e, Exception)
    prefix = _make_prefix(EXCEPTION, -3)
    lines = _make_exception(e, msg)
    for line in lines:
        _emit_line(prefix + line)


def colorful_exception(
    error=None, formatted=None, write_to_stderr=True, show_raised=True, compact=False
):
    accum = ""

    def s(*strs):
        nonlocal accum
        accum += "".join(strs) + reset

    tb_pat = re.compile(r"^.*File \"([^\"]+)\", line (\d+), in (.*)")

    def _traceback_match_filename(line):
        is_libs = False
        m = tb_pat.match(line)
        if m:
            file = m.group(1)
            lineno = m.group(2)
            context = m.group(3)
            real_path = os.path.realpath(file)
            relative_path = os.path.relpath(real_path)

            root = os.environ.get("ERISYON_ROOT")
            if root is not None:
                is_libs = True
                if real_path.startswith(root):
                    relative_path = re.sub(r".*/" + root, "./", real_path)
                    is_libs = False

            # Treat these long but commonly occurring path differently
            if "/site-packages/" in relative_path:
                relative_path = re.sub(r".*/site-packages/", ".../", relative_path)
            if "/dist-packages/" in relative_path:
                relative_path = re.sub(r".*/dist-packages/", ".../", relative_path)

            leading, basename = os.path.split(relative_path)
            # if leading and len(leading) > 0:
            #     leading = f"{'./' if leading[0] != '.' else ''}{leading}"
            return leading, basename, lineno, context, is_libs
        return None

    if not compact:
        s("\n")

    if formatted is None:
        formatted = traceback.format_exception(
            etype=type(error), value=error, tb=error.__traceback__
        )
    lines = []
    for line in formatted:
        lines += [sub_line for sub_line in line.strip().split("\n")]

    is_libs = False
    for line in lines[1:-1]:
        split_line = _traceback_match_filename(line)
        if split_line is None:
            s(gray if is_libs else "", line, "\n")
        else:
            leading, basename, lineno, context, is_libs = split_line
            if is_libs:
                s(gray, "File ", leading, "/", basename)
                s(gray, ":", lineno)
                s(gray, " in function ")
                s(gray, context, "\n")
            else:
                s("File ", yellow, leading, "/", yellow, bold, basename)
                s(":", yellow, bold, lineno)
                s(" in function ")
                s(magenta, bold, context, "\n")

    if show_raised:
        s(red, "raised: ", red, bold, error.__class__.__name__, "\n")
        error_message = str(error).strip()
        if error_message != "":
            s(red, error_message, "\n")

    if write_to_stderr:
        sys.stderr.write(accum)

    return accum


def current_file_and_line():
    frame = inspect.currentframe()
    try:
        context = inspect.getframeinfo(frame.f_back)
        return context.filename, context.lineno
    finally:
        del frame


def current_file_and_line_str():
    frame = inspect.currentframe()
    try:
        context = inspect.getframeinfo(frame.f_back)
        return f"{context.filename}:{context.lineno}"
    finally:
        del frame


def debug(*args):
    """
    Spew useful debug like...

        foo = 1
        bar = 2
        log.debug(foo, bar)
            # prints like: "fileno.py:10] foo=1 bar=2"

        log.debug(1+1)
            # prints like: "fileno.py:10] 1+1=2"

        log.debug('this is a message', foo)
            # prints like: "fileno.py:10] this is a message foo=1"

        log.debug()
            # prints like: "fileno.py:10] "
            # This is useful for just tracing progress during crash hunting

    Based on a hint from:
        https://stackoverflow.com/questions/8855974/find-the-name-of-a-python-variable-that-was-passed-to-a-function
    """
    frame = inspect.currentframe()
    try:
        values = []
        context = inspect.getframeinfo(frame.f_back)
        callers = "".join([line.strip() for line in context.code_context])
        m = re.search(r"debug\s*\((.+?)\)$", callers)
        if m:
            callers = [i.strip() for i in m.group(1).split(",")]
            for caller, arg in zip(callers, args):
                if isinstance(arg, str):
                    # debug('something', a, b)
                    # It may be a naked string
                    evaled = None
                    try:
                        evaled = eval(caller)
                    except Exception:
                        pass

                    if evaled == arg:
                        values += [arg]
                        continue

                if isinstance(arg, pd.core.frame.DataFrame):
                    values += [f"\n{green}{caller}{reset}=\n{arg}\n"]

                elif isinstance(arg, np.ndarray) or isinstance(
                    arg, pd.core.series.Series
                ):
                    values += [
                        f"\n{green}{caller}{reset}"
                        f"{yellow} {arg.dtype}{arg.shape}{reset}=\n{arg}\n"
                    ]

                else:
                    values += [
                        f"{green}{caller}{reset}"
                        f"{yellow}({_arg_type_str(arg)}){reset}={arg}"
                    ]

        msg = " ".join([str(i) for i in values])
        line = (
            f"\n{blue}{int(time.time())} {os.path.basename(context.filename)}:"
            f"{context.lineno}]{reset} {msg}\n"
        )

        _emit_line(line)

    finally:
        del frame


def prof(msg=""):
    """A quick and dirty profiler."""
    if not hasattr(thread_local, "prof_last_time"):
        thread_local.prof_last_time = None

    if thread_local.prof_last_time is None:
        elapsed = 0
    else:
        elapsed = time.time() - thread_local.prof_last_time

    frame = inspect.currentframe()
    try:
        context = inspect.getframeinfo(frame.f_back)
        line = (
            f"{blue}{elapsed:0.3f} {os.path.basename(context.filename)}:"
            f"{context.lineno}]{reset} {msg}\n"
        )

        _emit_line(line)

    finally:
        del frame
        thread_local.prof_last_time = time.time()


def input_request(message, default_when_headless):
    """
    Ask the user for input, but if headless return default_headless.
    If default_headless is an exception, then raise that.

    Note that this REQUIRES a default_headless argument so that you
    can not be lazy and avoid the question about what should happen in the
    case of a headless run.

    If this should never happen when headless, then pass in
    an Exception to the default_when_headless and it will be raised.
    """
    if is_headless():
        if isinstance(default_when_headless, Exception):
            raise default_when_headless
        return default_when_headless

    # Do not use the input(message) here because it doesn't wrap properly
    # in some cases (happened when I was using escape codes, but I didn't
    # bother to figure out why.)... Just print() works fine.
    info(message, strip=False, newline=False)
    return _input()


def confirm_yn(message, default_when_headless):
    """
    Prompt the user for an answer.
    See input_request() to understand why this requires a default_headless argument.
    """
    resp = None
    while resp not in ("y", "n"):
        resp = input_request(message + "(y/n): ", default_when_headless)
        resp = resp.lower()[0]
    return resp == "y"


if __name__ == "__main__":
    """
    TASK When used as a wrapper this module will launch a given sub-command and then
    convert all of its stdout standard formatting.
    """
    pass


def terminal_size():
    try:
        h, w, hp, wp = struct.unpack(
            "HHHH", fcntl.ioctl(0, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        )
        return w, h
    except OSError:
        # This can happen in a containerized context where no console exists
        # 80, 80 chosen as reasonable default w, h for imaginary terminal
        return 80, 80


def h_line(marker="-", label=""):
    count = (terminal_size()[0] - 1) // len(marker) - len(label)
    return label + marker * count