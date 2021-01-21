import os
import inspect
import time
from contextlib import contextmanager

# This will need to write out to seaparate files and then collate
# wchih means it needs temp access, etc.

I think this needs to be part of zap

@contextmanager
def profiler(**context_kws):
    """
    The context_kws are expected to be key=strings that are used for
    """
    assert all the context_kws are str val

    frame = inspect.currentframe()
    try:
        context = inspect.getframeinfo(frame.f_back)
        file = os.path.basename(context.filename)
        lineno = context.lineno
        start = time.time()
        yield
    finally:
        stop = time.time()
        _write()
