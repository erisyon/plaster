import gc
import sys
import time
from os.path import realpath

import numpy as np
from munch import Munch
from plaster.tools.log.log import colorful_exception, debug, info
from plaster.tools.utils import utils
from plumbum import colors, local


def timestamps_from_paths(paths, ignore_fn=None):
    """
    From a list of paths, recursively return the timestamps
    from all non-hidden and non-ignored paths

    args:
        A list of paths to recursively explore
        ignore_fn: Ignore paths where this callback returns True

    returns:
        a list of timestamps
    """

    if not isinstance(paths, list):
        paths = [paths]

    def keep(f):
        return f.name[0] != "." and not (ignore_fn is not None and ignore_fn(f))

    def recurse_times(path):
        times = []
        if path.exists() and keep(path):
            if path.is_dir():
                for f in path.list():
                    times += recurse_times(f)
            else:
                times += [(path, path.stat().st_mtime)]
        return times

    times = []
    for path in paths:
        times += recurse_times(path)

    return sorted(times, key=lambda i: i[0])


class PipelineState:
    # Ran and was successful
    success = "state_success.state"

    # Ran but but an error occurred
    error = "state_error.state"

    # An upstream dependency was in error state
    skipped = "state_skipped.state"

    # It would have run but the build limit option did not include it
    ignored = "state_ignore.state"

    # Nothing to do
    uptodate = "state_uptodate.state"

    all_states = (success, error, skipped, uptodate, ignored)
    success_states = (success, uptodate, ignored)
    error_states = (error, skipped)

    @staticmethod
    def get_state():
        """
        Find any state files or None if there aren't any in the current directory
        """
        for state in PipelineState.all_states:
            if local.path(state).exists():
                return state
        return None

    @staticmethod
    def set_state(state):
        # DELETE all other state files
        for _state in PipelineState.all_states:
            local.path(_state).delete()

        local.path(state).touch()

    @staticmethod
    def clear_error_state():
        # DELETE any error state files
        for _state in PipelineState.error_states:
            local.path(_state).delete()


class PipelineTask:
    """
    See class Pipeline docs below
    """

    @staticmethod
    def _config_dirty(new_config, old_config_path):
        if new_config is not None and old_config_path is not None:
            try:
                cur_config = utils.strip_underscore_keys(
                    utils.json_load_munch(old_config_path)
                )
                new_config = utils.strip_underscore_keys(new_config)
                if str(cur_config) != str(new_config):
                    return f"Config file changed"
            except FileNotFoundError:
                return "Never successfully run before"
            except Exception:
                return "Config not readable or parse-able"
        return None

    @staticmethod
    def _parent_timestamps(parents, ignore_fn):
        """mock-point"""
        return timestamps_from_paths(parents, ignore_fn)

    @staticmethod
    def _child_timestamps(children, ignore_fn):
        """mock-point"""
        return timestamps_from_paths(children, ignore_fn)

    @staticmethod
    def _out_of_date(parents, children, ignore_fn=None):
        """
        Check if parents are dirty compared to children

        args:
            parents: a list or singleton of paths.
                If the path is a dir, all of the files (recursively) in the dir will be used
            children: a list or singleton of paths.
                If the path is a dir, all of the files (recursively) in the dir will be used
            ignore_fn: ignore any path spec where this function return True

        return:
            A tuple: (out_of_date_boolean, reason)
            out_of_date_boolean: True if the youngest file in parents is younger than the oldest file in the children
            reason: The human readable reason why it is out of date
        """

        parent_files_and_times = PipelineTask._parent_timestamps(parents, ignore_fn)
        child_files_and_times = PipelineTask._child_timestamps(children, ignore_fn)

        if len(parent_files_and_times) == 0:
            return False, "No parent files"

        if len(child_files_and_times) == 0:
            return True, "No child files"

        parent_times = np.array(utils.listi(parent_files_and_times, 1))
        child_times = np.array(utils.listi(child_files_and_times, 1))

        if np.max(parent_times) > np.max(child_times):

            def name_fmt(path):
                path = local.path(path)
                return (
                    f"{utils.safe_list_get(path.split(), -2, default='')}/{path.name}"
                )

            parent_max_name = name_fmt(
                utils.listi(parent_files_and_times, 0)[np.argmax(parent_times)]
            )

            child_max_name = name_fmt(
                utils.listi(child_files_and_times, 0)[np.argmax(child_times)]
            )

            return (
                True,
                (
                    f"Parent file: '{parent_max_name}' "
                    f"is younger than child file: "
                    f"'{child_max_name}'"
                ),
            )

        return False, "Up to date"

    def __init__(
        self, src_dir, dst_dir, task_name, new_config, progress_fn=None, **kwargs
    ):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.task_name = task_name
        self.config = new_config
        self.progress_fn = progress_fn
        self.kwargs = kwargs
        self.dirty_reason = None
        self.wrote_state = False
        self.start_time = time.time()
        self.phase = (0, 1)
        with local.cwd(self.dst_dir):
            self.inputs = Munch.fromDict(
                {
                    k: self._resolve_path(v)
                    for k, v in (self.config.get("inputs") or {}).items()
                }
            )

    def _resolve_path(self, path):
        if path == "":
            return local.path(self.src_dir)
        else:
            if path.startswith("//"):
                return local.path(local.env["ERISYON_ROOT"]) / path[2:]
            else:
                return local.path(path)

    def set_phase(self, i, n_phases):
        self.phase = (i, n_phases)
        self.start_time = time.time()

    def progress(self, n_complete, n_total, retry):
        if retry:
            info(f"\nRetry {n_complete}")
        if self.progress_fn is not None:
            self.progress_fn(
                self.task_name, self.start_time, n_complete, n_total, self.phase
            )

    def start(self):
        """Called to start the task. Should return True is started, False if no need to run (not dirty)"""
        raise NotImplementedError

    def clear_error_state(self):
        with local.cwd(self.dst_dir):
            return PipelineState.clear_error_state()

    def get_output_state(self):
        with local.cwd(self.dst_dir):
            return PipelineState.get_state()

    def get_input_states(self):
        """
        Returns a dict of states for each input
        """
        found_states = Munch()
        for input_name, input_dir in self.inputs.items():
            try:
                with local.cwd(input_dir):
                    found_states[input_name] = PipelineState.get_state()
            except FileNotFoundError:
                found_states[input_name] = None
        return found_states

    def is_dirty(self, inputs=None):
        """
        Return dirty if:
            * config is dirty
            * Any non-state input is younger than then youngest non-state output
        """
        with local.cwd(self.dst_dir):
            config_dirty = self._config_dirty(self.config, "./config.json")
            if config_dirty is not None:
                self.dirty_reason = config_dirty
                return True

            if inputs is None:
                inputs = list(self.inputs.values())

            is_ood, reason = self._out_of_date(
                inputs,
                self.dst_dir,
                ignore_fn=lambda f: local.path(f).suffix == ".state"
                or local.path(f).name.startswith("_cache"),
            )
            if is_ood:
                self.dirty_reason = reason
                return True

            return False

    def success(self):
        """Write the success state without over-writing"""
        with local.cwd(self.dst_dir):
            if self.wrote_state:
                # Do not double write
                return

            self.wrote_state = True
            local.path(PipelineState.success).touch()
            utils.json_save(local.path("config.json"), self.config)

    def error(self, e=None):
        """Write the error state"""
        with local.cwd(self.dst_dir):
            if self.wrote_state:
                # Do not double write
                return

            self.wrote_state = True
            with open(local.path(PipelineState.error), "w") as f:
                f.write(str(e))


class Pipeline:
    """
    A single-threaded dependency graph pipeline evaluator.

    Inherit from the PipelineTask above for each type of task.

    Example usage:

        class MyTask(PipelineTask):
            def start(self):
                work_orders = [Munch(i=i) for i in range(5)]
                parallel_map(
                    do_some_work_func, work_orders, progress=self.progress
                )

        tasks = Munch(
            first_task=(MyTask, dict(root=root_dir), {}),
            second_task=(MyTask, dict(first_task=""), {}),
                # Note that the 'first_task=""' reference implies
                # that this task depends on first_task to be complete.
                # The dst_dir of the first_task will be available to
                # the seond_task as self.src_dirs.first_task
        )

        Pipeline(src_dir, dst_dir, tasks, options)

        See Pipeline.__init__() docs for more...
    """

    def _gc(self):
        """mock-point"""
        gc.collect()

    def _p(self, s):
        """mock-point"""
        sys.stdout.write(s)
        sys.stdout.flush()

    def _show_status(self, task_name, extra, terminator="\n", phase_str=""):
        self._p(
            f"{task_name:<{self.max_task_name_width}}: {phase_str}{extra}{terminator}"
        )

    def _progress(self, task_name, start_time, n_complete, n_total, phase=(0, 1)):
        """mock-point"""
        newline = "\n" if self.debug_mode else "\r"
        self._p(newline)

        phase_str = ""
        if phase[1] > 1:
            phase_str = f"(phase {phase[0] + 1} of {phase[1]}) "

        if n_total is None:
            # Unknown total
            status = f"{colors.green | str(n_complete)} of unknown"
        else:
            bar_length = 30
            done = int(bar_length * float(n_complete) / float(n_total))
            not_done = bar_length - done
            remain_char = "\u2591"
            done_char = "\u2587"
            bar = f"{done_char * done}{remain_char * not_done}"
            status = f"{colors.green | bar}"

        if n_complete > 0:
            elapsed = time.time() - start_time
            remain = (elapsed / n_complete) * (n_total - n_complete)

            status += (
                f" elapsed: {utils.elapsed_time_in_minutes_seconds(elapsed)},"
                f" ~remain: {utils.elapsed_time_in_minutes_seconds(remain)}   "
            )
        self._show_status(task_name, status, terminator="", phase_str=phase_str)

    def _log(self, task_name, state, message=None):
        self._logs += [(time.time(), task_name, state, message)]

    def logs(self):
        return self._logs

    def failed_count(self):
        return self._failed_count

    def __init__(
        self,
        src_dir,
        dst_dir,
        tasks,
        debug_mode=False,
        force=False,
        limit=None,
        clean=False,
    ):
        """
        args:
            src_dir: Where the source files live for the project
            dst_dir: When the task outputs will go
                     (a directory will be created for each task in dst_dir)
            tasks: A dict of task-tuples like:
                {
                    "task_a": (TaskA, config_for_task_a, kwargs_for_task_a)
                }
                Where config_for_task_X is a dict like:
                {
                    "inputs": {
                        "source": "",
                        "a": "../a"
                    },
                    "parameters": {}
                }
            debug_mode: If True, new lines are added to progress to ease reading logs
            force: If True, targets are rebuilt even if they are up-to-date
            limit: An optional list of targets to build (no other targets will be built)
            clean: If True, clean the outputs and exit immediately
        """

        self.tasks = tasks
        self.src_dir = local.path(src_dir)
        self.dst_dir = local.path(dst_dir)
        self._failed_count = 0
        self._logs = []

        # Find the max task name for good formatting
        self.max_task_name_width = max([len(task_name) for task_name in tasks.keys()])

        # When debug_mode is set new lines are added in progress so that
        # it is easier to read debugging logs.
        self.debug_mode = debug_mode

        root_folder = local.cwd

        # VALIDATE the limit target names
        if limit is not None:
            for target in limit:
                if target not in tasks.keys():
                    self._p(f"{target}: {colors.red | 'Unknown target'}\n")
                    return

        # SETUP task list permutations
        all_tasks = list(tasks.keys())
        limit_tasks = limit
        allow_tasks = limit_tasks or all_tasks
        ignored_tasks = [
            task for task in all_tasks if limit_tasks and task not in limit_tasks
        ]

        # MKDIR for all targets
        for task_name in all_tasks:
            (self.dst_dir / task_name).mkdir()

        task_munches = {
            task_name: Munch(
                target_dir=dst_dir / task_name,
                klass=task_tuple[0],
                config=task_tuple[1],
                kwargs=task_tuple[2],
                complete=False,
                ignore=task_name in ignored_tasks,
                task=None,
            )
            for task_name, task_tuple in tasks.items()
        }

        task_munch_by_target_dir = {
            realpath(str(t.target_dir)): t for _, t in task_munches.items()
        }

        # INSTANTIATE task for each target and clear any error states
        for task_name, task_munch in task_munches.items():
            task_munch.task = task_munch.klass(
                self.src_dir,
                self.dst_dir / task_name,
                task_name,
                task_munch.config,
                progress_fn=self._progress,
                **task_munch.kwargs,
            )
            task_munch.task.clear_error_state()

            # CLEAN if requested (only allowed targets)
            if clean and task_name in allow_tasks:
                with local.cwd(self.dst_dir / task_name):
                    for f in self.dst_dir / task_name:
                        f.delete()
                self._show_status(task_name, f"{colors.green | 'Cleaned'}")

        # Early out on a clean request
        if clean:
            return

        # RUN until all complete (complete includes error, up to date, etc)
        while True:
            # SCAN for anything to do
            for task_name, task_munch in task_munches.items():

                def started(reason):
                    self._log(task_name, "started")
                    self._show_status(
                        task_name, f"{colors.green | 'Started'} (Reason: {reason})"
                    )

                def uptodate():
                    self._log(task_name, "uptodate")
                    self._show_status(task_name, f"{colors.green | 'Up to date'}")

                def success():
                    self._log(task_name, "success")
                    self._progress(task_name, task_munch.task.start_time, 100, 100)
                    self._p("\n")
                    self._show_status(task_name, f"{colors.green | 'Success'}")

                def skipped(upstream_failures):
                    self._log(task_name, "skipped")
                    self._show_status(
                        task_name,
                        f"{colors.yellow | 'Skipped due to upstream failure(s)'}: {','.join(upstream_failures)}",
                    )

                def ignored():
                    self._log(task_name, "ignored")
                    self._show_status(
                        task_name,
                        f"{colors.yellow | 'Ignored because it was not in the limit list'}",
                    )

                def failed(e):
                    self._log(task_name, "failed", f"{e.__class__.__name__}: {e}")
                    self._failed_count += 1
                    self._p("\n")
                    self._show_status(
                        task_name,
                        f"{colors.red | 'Failed'} {e.__class__.__name__}: {e}",
                    )

                    if not hasattr(e, "ignore_traceback"):
                        line = "- " * (utils.terminal_size()[0] // 2)
                        self._p(f"{colors.red | line}\n")
                        try:
                            if hasattr(e, "stderr"):
                                self._p(colors.yellow | ("".join(e.stderr)))

                            if hasattr(e, "exception_lines"):
                                # Traceback came from a sub-process
                                lines = colorful_exception(
                                    formatted=e.exception_lines,
                                    write_to_stderr=False,
                                    show_raised=False,
                                    compact=True,
                                )
                            else:
                                with local.cwd(root_folder):
                                    lines = colorful_exception(
                                        e,
                                        write_to_stderr=False,
                                        show_raised=False,
                                        compact=True,
                                    )
                            self._p(lines)
                        except Exception as _:
                            self._p(colors.red | f"\nUnknown exception: {e}\n")

                        self._p(f"{colors.red | line}\n\n")

                with local.cwd(dst_dir / task_name):
                    # CONSIDER starting tasks if its dependencies are complete
                    if not task_munch.complete:
                        # CHECK if it is being ignored
                        if task_munch.ignore:
                            ignored()
                            task_munch.complete = True

                        else:
                            # FETCH all the upstream states
                            upstream_states = task_munch.task.get_input_states()
                            upstream_failures = [
                                task
                                for task, state in upstream_states.items()
                                if state in PipelineState.error_states
                            ]

                            # SKIP this task if there are any upstream failure
                            if len(upstream_failures) > 0:
                                PipelineState.set_state(PipelineState.skipped)
                                skipped(upstream_failures)
                                task_munch.complete = True

                            else:
                                # No upstream errors... CHECK if all upstream are complete
                                # by looking at complete flag inside task_munch. We need to
                                # look up the upstream tasks by their output dir since
                                # we only know their local key value names for input items,
                                # which are not necessarily the same as PipelineTask names.

                                upstream_dirs = [
                                    input_dir
                                    for _, input_dir in task_munch.task.inputs.items()
                                    if input_dir in task_munch_by_target_dir
                                ]

                                input_tasks_complete = [
                                    task_munch_by_target_dir[upstream_dir].complete
                                    for upstream_dir in upstream_dirs
                                ]

                                all_inputs_ready = all(input_tasks_complete)

                                if all_inputs_ready:
                                    # Ready to build, no matter what happens now this task is considered complete
                                    task_munch.complete = True
                                    try:
                                        if task_munch.task.is_dirty() or force:
                                            started(
                                                "Forced"
                                                if force
                                                else task_munch.task.dirty_reason
                                            )
                                            task_munch.task.start_time = time.time()
                                            task_munch.task.progress(0, 1, False)

                                            # I've considered putting a "rm *" here because
                                            # really it should clear any bad results from a
                                            # previous run but that scares me a little bit
                                            # because if the logic of changing correct directory
                                            # failed somehow then there would be a rogue delete
                                            # For now, I'm just not going to do this.

                                            task_munch.task.start()
                                            self._gc()
                                            PipelineState.set_state(
                                                PipelineState.success
                                            )
                                            task_munch.task.success()
                                            success()
                                        else:
                                            # Mark as up to date if not dirty and it previously succeeded
                                            PipelineState.set_state(
                                                PipelineState.uptodate
                                            )
                                            uptodate()
                                    except Exception as e:
                                        if not hasattr(e, "exc_type"):
                                            (
                                                e.exc_type,
                                                e.exc_value,
                                                e.exc_traceback,
                                            ) = sys.exc_info()
                                        PipelineState.set_state(PipelineState.error)
                                        failed(e)
                                        task_munch.task.error(e)

            all_done = all([i.complete for i in task_munches.values()])
            if all_done:
                break

            time.sleep(0.01)
