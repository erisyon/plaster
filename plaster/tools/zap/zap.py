"""
This is a wrapper for executing various parallel map
routines in a consistent way.

The three primary routines are:
    work_orders()
    arrays()
    df_rows()
    df_groups()


Debugging run-away processes.

    Sometimes you can get into a situation where processes seem
    to be stranded and are still running after a ^C.
    This is complicated by running docker under OSX.
    Docker under OSX is actually running under a Linux VM called
    "com.docker.hyperkit". The OSX pid of that process
    has nothing to do with the pid of the processes that are
    running inside the VM and the pids running insider the
    container (inside the VM).

    You can drop into the hyperkit VM with this command form OSX:
        $ OSX_VM=1 ./p

    Once in there you can "top" and see what processes are
    running. Let's say that you see that pid 5517 taking 100% cpu.
    You can then find the pid INSIDE the container with this:
        $ cat /proc/5517/status | grep NSpid
        > NSpid:	5517	832

    The second number of which is the pid INSIDE the container (832).

"""
import os
import random
import signal
import sys
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    thread,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from munch import Munch
from plaster.tools.log.log import debug, error, exception, info, prof
from plaster.tools.utils import utils

global_cpu_limit = None
global_debug_mode = None
global_progress = None


if os.environ.get("ZAP_DEBUG_MODE") == "True":
    global_debug_mode = True


@contextmanager
def Context(cpu_limit=None, debug_mode=None, progress=None, prevent_inner_parallelism=True):
    """
    When prevent_inner_parallelism is True, a Context that is created inside of
    """
    global global_cpu_limit, global_debug_mode, global_progress
    orig_cpu_limit = global_cpu_limit
    orig_debug_mode = global_debug_mode
    orig_progress = global_progress
    global_cpu_limit = cpu_limit
    global_debug_mode = debug_mode
    global_progress = progress
    yield
    global_cpu_limit = orig_cpu_limit
    global_debug_mode = orig_debug_mode
    global_progress = orig_progress


def _cpu_count():
    """mock-point"""
    return cpu_count()


def _show_work_order_exception(e):
    """Mock-point"""
    error("\nAn exception was thrown by a work_order ------")
    info("".join(e.exception_lines))
    error("----------------------------------------------")


def _mock_BrokenProcessPool_exception():
    """mock_point"""
    pass


def wtf():
    print(f"pid=", os.getpid())
    for k, v in globals().items():
        if k.startswith("__zap"):
            print(k, v)


def _set_zap(**kwargs):
    """
    Creates a global variable with the zap information to bypass
    the serialization that multiprocessing would otherwise do.
    """
    zap_id = int(time.time() * 10000)
    zap = Munch(id=zap_id, **kwargs)
    globals()[f"__zap_{zap_id}"] = zap
    return zap


def _get_zap(zap_id):
    """
    Fetches zap data from global. See _set_zap.
    """
    return globals()[f"__zap_{zap_id}"]


def _del_zap(zap_id):
    del globals()[f"__zap_{zap_id}"]


def _run_work_order_fn(zap_id, work_order_i):
    """
    Wrap the function to handle args, kwargs, capture exceptions, and re-seed RNG.
    Note: This may run in the sub-process or thread and therefore should not use stdio.
    """
    start_time = time.time()
    try:
        work_order = _get_zap(zap_id).work_orders[work_order_i]

        # RE-INITIALIZE the random seed because numpy resets the seed in sub-processes.
        np.random.seed(seed=int(time.time() * 100_000) % int(2 ** 32))
        random.seed()

        args = work_order.pop("args", ())
        fn = work_order.pop("fn")
        assert callable(fn)
        result = fn(*args, **work_order)
    except Exception as e:
        formatted = traceback.format_exception(
            etype=type(e), value=e, tb=e.__traceback__
        )
        result = e
        result.exception_lines = formatted

    return result, time.time() - start_time


def _call_progress(zap, i, retry=False):
    if zap.progress is not None:
        try:
            zap.progress(i + 1, zap.n_work_orders, retry)
        except Exception as e:
            exception(e, "Warning: progress function exceptioned; ignoring.")


def _examine_result(zap, result, work_order):
    if isinstance(result, Exception):
        result.work_order = work_order
        if not zap.trap_exceptions:
            raise result
    return result


def _do_zap_with_executor(executor, zap):
    """
    Execute work_orders through a thread or process pool executor
    """
    retry_iz = []

    wo_i_by_future = {}
    for i, work_order in enumerate(zap.work_orders):
        # Important: the executor submit must not be passed
        # the actual work_order to bypass serialization.
        future = executor.submit(_run_work_order_fn, zap.id, i)
        wo_i_by_future[future] = i

    results = [None] * zap.n_work_orders
    timings = [None] * zap.n_work_orders

    n_done = 0
    for future in as_completed(wo_i_by_future):
        i = wo_i_by_future[future]
        work_order = zap.work_orders[i]
        try:
            result, duration = future.result()
            _mock_BrokenProcessPool_exception()  # Used for testing
            _call_progress(zap, n_done)
            n_done += 1
            results[i] = _examine_result(zap, result, work_order)
            timings[i] = duration
        except BrokenProcessPool:
            # This can happen if the child process(es) run out
            # of memory. In that case, we need to retry those
            # work_orders.
            retry_iz += [i]

    for i in retry_iz:
        # These retries are likely a result of running out memory
        # and we don't know how many of those processes we can support
        # so the only safe thing to do is to run them one at a time.
        # If this becomes a constant issue then we could try some
        # sort of exponential backoff on number of concurrent processes.
        # Sometimes this can create a queue.Full exception in the babysitter
        # threads that are not handled gracefully by Python.
        # See:
        # https://bugs.python.org/issue8426
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
        # https://github.com/python/cpython/pull/3895
        try:
            _call_progress(zap, i, retry=True)
            result, duration = _run_work_order_fn(zap.id, i)
            results[i] = _examine_result(zap, result, zap.work_orders[i])
            timings[i] = duration
        except Exception as e:
            results[i] = e
            timings[i] = None

    return results, timings


def _do_work_orders_process_mode(zap):
    if sys.platform != "linux":
        raise Exception(
            "Process mode zap is not working under non-linux at moment sue to differences in globals() under non-linux."
        )

    with ProcessPoolExecutor(max_workers=zap.max_workers) as executor:
        try:
            return _do_zap_with_executor(executor, zap)
        except KeyboardInterrupt:
            # If I do not os.kill the processes then it seems
            # that will gracefully send kill signals and wait
            # for the children. I typically want to just abandon
            # anything that the process is doing and have it end instantly.
            # Thus, I just reach in to get the child pids and kill -9 them.
            for k, v in executor._processes.items():
                try:
                    os.kill(v.pid, signal.SIGKILL)
                except ProcessLookupError:
                    info(f"{v.pid} had already died")
            raise


def _do_work_orders_thread_mode(zap):
    with ThreadPoolExecutor(
        max_workers=zap.max_workers, thread_name_prefix=zap.thread_name_prefix
    ) as executor:
        try:
            return _do_zap_with_executor(executor, zap)
        except BaseException as e:
            # Any sort of exception needs to clear all threads.
            # Note that KeyboardInterrupt inherits from BaseException not
            # Exception so using BaseException to include KeyboardInterrupts
            # Unlike above with os.kill(), the thread clears are not so destructive,
            # so we want to call them in any situation in which we're bubbling up the
            # exception.
            executor._threads.clear()
            thread._threads_queues.clear()
            raise e


def _do_work_orders_debug_mode(zap):
    """
    debug_mode skips all multi-processing so that console-based debuggers are happy
    """
    results = [None] * zap.n_work_orders
    timings = [None] * zap.n_work_orders
    for i, work_order in enumerate(zap.work_orders):
        result, duration = _run_work_order_fn(zap.id, i)
        results[i] = _examine_result(zap, result, work_order)
        timings[i] = duration
        _call_progress(zap, i)

    return results, timings


def get_cpu_limit(_cpu_limit=None):
    if _cpu_limit is None:
        _cpu_limit = global_cpu_limit

    if _cpu_limit is None:
        _cpu_limit = _cpu_count()

    if _cpu_limit < 0:
        _cpu_limit = _cpu_count() + _cpu_limit + 1  # eg: 4 cpu + (-1) + 1 is 4

    assert _cpu_limit > 0
    return _cpu_limit


def work_orders(
    _work_orders,
    _process_mode=True,
    _trap_exceptions=True,
    _thread_name_prefix="",
    _cpu_limit=None,
    _debug_mode=None,
    _progress=None,
    _return_timings=False,
):
    """
    Runs work_orders in parallel.

    work_orders: List[Dict]
        Each work_order should have a "fn" element that points to the fn to run
        If the work_order has an "args" element those will be passed as *args
        all other elements of the work_order will be passed as **kwargs
    _process_mode:
        If True, parallelize via sub-processes bypassing the GIL.
    _trap_exceptions:
        If True: a work_order that exceptions will be trapped
        and the exception will be returned as that work_order's result.
        If False: any exception in a work_order will immediately
        bubble and cancel all other work_orders.
    _thread_name_prefix:
        Add a prefix is thread mode
    _cpu_limit:
        If -1, use all cpus, -2 all but one, etc.
    _debug_mode:
        If True, bypass all multi-processing and run each work_order serially.
        This is useful if you need to use pudb or similar.
    _progress:
        If not None, expected to be callable like:
            process(i, j, retry)
        Where:
            i is how many work_orders have completed,
            j is the total number of work_orders
            retry is False normally and will be True if a particular
            work_order (passed as i) had to be retried due to
            memory failure.
    _return_timings:
        If True, then returns a tuple of results, timings
        otherwise just returns results

    Note that many of these arguments can be created with a Context like:

    with zap.Context(cpu_limit=2):
        zap.work_orders(work_orders)
    """
    if _debug_mode is None:
        _debug_mode = global_debug_mode

    if _debug_mode is None:
        _debug_mode = False

    if _progress is None:
        _progress = global_progress

    _cpu_limit = get_cpu_limit(_cpu_limit)

    zap = _set_zap(
        work_orders=_work_orders,
        n_work_orders=len(_work_orders),
        progress=_progress,
        thread_name_prefix=_thread_name_prefix,
        trap_exceptions=_trap_exceptions,
        max_workers=_cpu_limit,
    )

    try:
        if _debug_mode:
            # debug_mode takes precedence; ie over-rides any multi-processing
            results, timings = _do_work_orders_debug_mode(zap)
        elif _process_mode:
            results, timings = _do_work_orders_process_mode(zap)
        else:
            results, timings = _do_work_orders_thread_mode(zap)
    except Exception as e:
        if hasattr(e, "exception_lines"):
            _show_work_order_exception(e)
        raise e
    finally:
        _del_zap(zap.id)

    if _return_timings:
        return results, timings
    return results


def make_batch_slices(n_rows, _batch_size=None, _limit_slice=None):
    if _limit_slice is None:
        _limit_slice = slice(0, n_rows, 1)

    if isinstance(_limit_slice, int):
        _limit_slice = slice(0, _limit_slice, 1)

    _limit_slice = [_limit_slice.start, _limit_slice.stop, _limit_slice.step]

    if _limit_slice[2] is None:
        _limit_slice[2] = 1

    if _limit_slice[1] is None:
        _limit_slice[1] = n_rows

    assert _limit_slice[2] == 1  # Until I have time to think this through
    n_rows = _limit_slice[1] - _limit_slice[0]

    if _batch_size is None:
        # If not specified, base it on the number of cpus.
        # Note, if n_batches is only as big as the _cpu_count then there won't
        # be any output on the progress bar until it is done so it is scaled
        # by eight here to ensure the progress bar will at least move 8 times.
        n_batches = min(n_rows, 8 * _cpu_count())
        batch_size = max(1, (n_rows // n_batches) + 1)
    else:
        batch_size = _batch_size
        n_batches = max(
            1, (n_rows // batch_size) + (0 if n_rows % batch_size == 0 else 1)
        )

    if batch_size <= 0:
        raise ValueError(f"illegal batch_size {batch_size}")

    assert batch_size * n_batches >= n_rows

    batch_slices = []
    for batch_i in range(n_batches):
        start = _limit_slice[0] + batch_i * batch_size
        stop = _limit_slice[0] + min((batch_i + 1) * batch_size, n_rows)
        if stop > start:
            batch_slices += [(start, stop)]
    return batch_slices


def _run_arrays(inner_fn, slice, arrays_dict, **kwargs):
    """
    Assumes that the lengths of the value arrays are all the same.
    """
    # SETUP the re-usable kwargs with parameters and arrays and then poke values one row at a time
    res = []
    for row_i in range(slice[0], slice[1]):
        for field_i, (key, array) in enumerate(arrays_dict.items()):
            kwargs[key] = array[row_i]
        val = inner_fn(**kwargs)
        if isinstance(val, tuple):
            res += [val]
        else:
            res += [(val,)]

    return res


def arrays(
    fn,
    arrays_dict,
    _batch_size=None,
    _stack=False,
    _limit_slice=None,
    _process_mode=True,
    _thread_name_prefix="",
    _debug_mode=None,
    _progress=None,
    _trap_exceptions=False,
    **kwargs,
):
    """
    Split an array by its first dimension and send each row to fn.
    The array_dict is one or more parallel arrays that will
    be passed to fn(). **kwargs will end up as (constant) kwargs
    to fn().

    Example:
        def myfn(a, b, c):
            return a + b + c

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        res = zap.arrays(
            myfn,
            dict(a=a, b=b),
            c=1
        )

        # This will call:
        #   myfn(1, 4, 1)
        #   myfn(2, 5, 1)
        #   myfn(3, 6, 1)
        # and res == [1+4+1, 2+5+1, 3+6+1]

    These calls are batched into parallel processes (or _process_mode is False)
    where the _batch_size is set or if None it will be chosen to use all cpus.

    When fn returns a tuple of fields, these return fields
    will be maintained.

    Example:
        def myfn(a, b, c):
            return a, b+c

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        res = zap.arrays(
            myfn,
            dict(a=a, b=b),
            c=1
        )

        # This will call as before but now:
        #   res == ([1, 2, 3], [4+1, 5+1, 6+1])

    If _stack is True then _each return field_ will be wrapped
    with a np.array() before it is returned.  If _stack is a list
    then you can selective wrap the np.array only to the return
    fields of your choice.

    Example:
        def myfn(a, b, c):
            return a, b+c

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        res = zap.arrays(
            myfn,
            dict(a=a, b=b),
            c=1,
            _stack=True
        )

        # This will call as before but now:
        #   res == (np.array([1, 2, 3]), np.array([4+1, 5+1, 6+1]))
        # Of called with _stack=[True, False]
        #   res == (np.array([1, 2, 3]), [4+1, 5+1, 6+1])
    """

    n_rows = len(list(arrays_dict.values())[0])
    assert all([len(a) == n_rows for a in arrays_dict.values()])

    batch_slices = make_batch_slices(n_rows, _batch_size, _limit_slice)

    result_batches = work_orders(
        _work_orders=[
            Munch(
                fn=_run_arrays,
                inner_fn=fn,
                slice=batch_slice,
                arrays_dict=arrays_dict,
                **kwargs,
            )
            for batch_slice in batch_slices
        ],
        _trap_exceptions=_trap_exceptions,
        _process_mode=_process_mode,
        _progress=_progress,
        _thread_name_prefix=_thread_name_prefix,
        _debug_mode=_debug_mode,
    )

    if len(result_batches) == 0:
        raise ValueError("No batches were returned")
    first_batch = result_batches[0]
    if isinstance(first_batch, Exception):
        raise first_batch
    if len(first_batch) == 0:
        raise ValueError("First batch had no elements")
    first_return = first_batch[0]
    if isinstance(first_return, Exception):
        raise first_return

    assert isinstance(first_return, tuple)
    n_fields = len(first_return)

    unbatched = []
    for field_i in range(n_fields):
        field_rows = []
        for batch in result_batches:
            field_rows += utils.listi(batch, field_i)
        unbatched += [field_rows]

    if _stack is not None:
        if isinstance(_stack, bool):
            _stack = [_stack] * n_fields

        if isinstance(_stack, (list, tuple)):
            assert all([isinstance(s, bool) for s in _stack])
            assert len(_stack) == n_fields

        # If requested, wrap the return field in np.array()
        for field_i in range(n_fields):
            if _stack[field_i]:
                unbatched[field_i] = np.array(unbatched[field_i])

    if n_fields == 1:
        return unbatched[0]
    else:
        return tuple(unbatched)


def _run_df_rows(inner_fn, slice, df, **kwargs):
    """
    Assumes that the lengths of the value arrays are all the same.
    """
    # SETUP the re-usable kwargs with parameters and arrays and then poke values one row at a time
    res = []
    for row_i in range(slice[0], slice[1]):
        args = (df.iloc[row_i : row_i + 1],)
        val = inner_fn(*args, **kwargs)
        res += [val]

    return res


def df_rows(
    fn,
    df,
    _batch_size=None,
    _limit_slice=None,
    _process_mode=True,
    _thread_name_prefix="",
    _debug_mode=None,
    _progress=None,
    **kwargs,
):
    """
    Split a dataframe along its rows. I do not want to actually
    split it because I want to minimize what is serialized.
    """
    n_rows = len(df)

    batch_slices = make_batch_slices(n_rows, _batch_size, _limit_slice)

    result_batches = work_orders(
        _work_orders=[
            Munch(fn=_run_df_rows, inner_fn=fn, slice=batch_slice, df=df, **kwargs,)
            for batch_slice in batch_slices
        ],
        _trap_exceptions=False,
        _process_mode=_process_mode,
        _progress=_progress,
        _thread_name_prefix=_thread_name_prefix,
        _debug_mode=_debug_mode,
    )

    unbatched = []
    for batch in result_batches:
        for ret in batch:
            if not isinstance(ret, pd.DataFrame):
                raise TypeError(
                    "return values from the fn of df_rows must be DataFrames"
                )
            unbatched += [ret]
    return pd.concat(unbatched).reset_index(drop=True)


def df_groups(fn, df_group, **kwargs):
    """
    Run function on each group of groupby

    There is a lot of complexity to the way that groupby handles return
    values from the functions so I use the apply to accumlate the
    work orders and then use apply again to return the results and
    let the apply whatever magic it wants to to reformat the result
    """
    # kwargs.pop("_progress")
    # kwargs.pop("_trap_exceptions")
    # kwargs.pop("_process_mode")
    # return pd_group.apply(fn, **kwargs)

    def _do_get_calls(group, **kwargs):
        # CONVERT the index to a tuple so that it can be hashed
        # There's a bug that shows up in pandas 1.1.0 with the following copy
        # ZBS didn't have time to figure out if there's a work around
        # so I pinned Pandas back to 1.0.5
        return Munch(args=(group.copy(),), _index=tuple(group.index.values), **kwargs)

    _work_orders = df_group.apply(_do_get_calls)

    wo_kwargs = {}
    non_wo_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith("_"):
            non_wo_kwargs[k] = v
        else:
            wo_kwargs[k] = v

    _work_orders_with_fn = []
    for wo in _work_orders:
        del wo["_index"]
        wo["fn"] = fn
        wo.update(wo_kwargs)
        _work_orders_with_fn += [wo]

    results = work_orders(_work_orders_with_fn, **non_wo_kwargs)

    # results is a list. One element per work order which in this case is
    # one work_order per group.
    #
    # Each WO result is the return value of the function; if multiple
    # return values then it is a tuple.

    return results
