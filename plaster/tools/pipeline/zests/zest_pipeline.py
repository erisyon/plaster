import time
from plumbum import local
from plaster.tools.utils import utils
from munch import Munch
from plaster.tools.pipeline.pipeline import PipelineTask, PipelineState, Pipeline
from zest import zest


def zest_PipelineTask_helpers():
    def config_dirty():
        def it_returns_none_on_new_config_missing():
            assert PipelineTask._config_dirty(None, "test") is None

        def it_returns_none_on_old_config_missing():
            assert PipelineTask._config_dirty("test", None) is None

        def it_returns_none_if_old_and_new_identical():
            new_config = Munch(a=1, b=2)
            utils.json_save("/tmp/config.json", new_config)
            assert PipelineTask._config_dirty(new_config, "/tmp/config.json") is None

        def it_returns_non_none_if_old_and_new_different():
            new_config = Munch(a=1, b=2)
            utils.json_save("/tmp/config.json", new_config)
            new_config["c"] = 3
            assert (
                PipelineTask._config_dirty(new_config, "/tmp/config.json") is not None
            )

        def it_returns_none_if_old_and_new_differ_only_by_underscore_blocks():
            new_config = Munch(a=1, b=2, _c=3)
            utils.json_save("/tmp/config.json", new_config)
            new_config["_c"] = 4
            assert PipelineTask._config_dirty(new_config, "/tmp/config.json") is None

        def it_returns_non_none_if_old_not_found():
            assert (
                PipelineTask._config_dirty(Munch(a=1), "/tmp/doesnt_exist_config.json")
                is not None
            )

        def it_returns_non_none_if_old_un_parseable():
            with open("/tmp/config.json", "w") as f:
                f.write("This is not json")
            assert (
                PipelineTask._config_dirty(Munch(a=1), "/tmp/config.json") is not None
            )

        zest()

    def out_of_date():
        m_parent_timestamps = zest.stack_mock(PipelineTask._parent_timestamps)
        m_child_timestamps = zest.stack_mock(PipelineTask._child_timestamps)

        def it_returns_false_on_no_parent_files():
            m_parent_timestamps.returns([])
            assert PipelineTask._out_of_date("parent", "child")[0] is False

        def it_returns_true_on_no_child_files():
            m_parent_timestamps.returns([("a0", 1)])
            m_child_timestamps.returns([])
            assert PipelineTask._out_of_date("parent", "child")[0] is True

        def it_returns_true_if_any_parent_file_is_younger_than_youngest_child():
            m_parent_timestamps.returns([("p0", 3)])
            m_child_timestamps.returns([("c0", 1), ("c1", 2)])
            assert PipelineTask._out_of_date("parent", "child")[0] is True

        def it_returns_false_if_all_parent_files_are_older_than_all_child_files():
            m_parent_timestamps.returns([("p0", 1)])
            m_child_timestamps.returns([("c0", 2), ("c1", 3)])
            assert PipelineTask._out_of_date("parent", "child")[0] is False

        zest()

    zest()


def zest_PipelineTask():
    task_name, src_dir, dst_dir = None, None, None

    def _before():
        nonlocal task_name, src_dir, dst_dir
        task_name = "test"

        # use 'with' due to bug in plumbum in which symlinks are only expanded using 'with' context,
        # and on osx /tmp is a symlink to /private/tmp
        with local.cwd("/tmp"):
            src_dir = local.path("./test_pipeline")
            src_dir.delete()
            src_dir.mkdir()
            dst_dir = src_dir / "output" / task_name
            dst_dir.delete()
            dst_dir.mkdir()

    def _pt(config={}, progress_fn=None):
        return PipelineTask(src_dir, dst_dir, task_name, config, progress_fn)

    def it_expands_empty_input_to_src_dir():
        pt = _pt(Munch(inputs=Munch(a="")))
        assert pt.inputs.a == src_dir

    def it_expands_relative_input_inside_dst_dir():
        pt = _pt(Munch(inputs=Munch(a="../a")))
        assert pt.inputs.a == src_dir / "output" / "a"
        assert pt.inputs.a == dst_dir / "../a"

    def it_bubbles_progress():
        was_called = False

        def _call_me(task_name, start_time, n_complete, n_total, phase=()):
            nonlocal was_called
            assert start_time > 0
            assert task_name == task_name and n_complete == 1 and n_total == 1
            was_called = True

        pt = _pt(Munch(inputs=Munch(a="../a")), progress_fn=_call_me)
        pt.progress(1, 1, retry=False)
        assert was_called

    def it_returns_dirty_check_config():
        with zest.mock(PipelineTask._config_dirty, returns="config dirty"):
            pt = _pt(Munch(inputs=Munch(a="../a")))
            assert pt.is_dirty() is True
            assert pt.dirty_reason == "config dirty"

    def it_uses_inputs_args_if_specified():
        with zest.mock(PipelineTask._config_dirty, returns=None) as m_config_dirty:
            with zest.mock(
                PipelineTask._out_of_date, returns=(True, "because, reasons")
            ) as m_out_of_date:
                pt = _pt(Munch(inputs=Munch(a="../a")))
                inputs = [src_dir / "a", src_dir / "b"]
                pt.is_dirty(inputs)
                assert m_out_of_date.n_calls == 1
                calls = m_out_of_date.normalized_calls()[0]
                assert calls["parents"] == inputs
                assert calls["children"] == dst_dir

    def it_writes_succcess_state():
        success_file = dst_dir / PipelineState.success
        assert not success_file.exists()
        pt = _pt()
        pt.success()
        assert success_file.exists()

    def it_does_not_writes_succcess_state_twice():
        success_file = dst_dir / PipelineState.success
        assert not success_file.exists()
        pt = _pt()
        pt.success()
        assert success_file.exists()
        success_file.delete()
        pt.success()
        assert not success_file.exists()

    def it_writes_succcess_with_config():
        pt = _pt(Munch(a=1))
        pt.success()
        new_config = utils.json_load(dst_dir / "config.json")
        assert new_config == Munch(a=1)

    def it_writes_error_state():
        error_file = dst_dir / PipelineState.error
        assert not error_file.exists()
        pt = _pt()
        pt.error()
        assert error_file.exists()

    def it_does_not_write_error_state_twice():
        error_file = dst_dir / PipelineState.error
        assert not error_file.exists()
        pt = _pt()
        pt.error()
        assert error_file.exists()
        error_file.delete()
        pt.error()
        assert not error_file.exists()

    def it_gets_output_state_if_it_exists():
        error_file = dst_dir / PipelineState.error
        error_file.touch()
        pt = _pt()
        assert pt.get_output_state() == PipelineState.error

    def it_gets_output_state_none_if_it_does_not_exists():
        pt = _pt()
        assert pt.get_output_state() is None

    def it_gets_all_inputs_states():
        a = dst_dir / "../a"
        a.mkdir()

        b = dst_dir / "../b"
        b.mkdir()

        (a / PipelineState.success).touch()
        (b / PipelineState.error).touch()

        pt = _pt(Munch(inputs=Munch(a="../a", b="../b")))
        states = pt.get_input_states()
        assert states.a == PipelineState.success
        assert states.b == PipelineState.error

    def it_gets_non_existant_inputs_states():
        a = dst_dir / "../a"
        assert not a.exists()
        pt = _pt(Munch(inputs=Munch(a=a)))
        states = pt.get_input_states()
        assert states.a is None

    def it_ignored_src_states():
        pt = _pt(Munch(inputs=Munch(a="")))
        states = pt.get_input_states()
        assert states.a is None

    zest()


class Task1(PipelineTask):
    def start(self):
        pass

    def is_dirty(self, inputs=None):
        pass


class Task2(PipelineTask):
    def start(self):
        pass

    def is_dirty(self, inputs=None):
        pass


class Task3(PipelineTask):
    def start(self):
        pass

    def is_dirty(self, inputs=None):
        pass


"""
t_clears_old_state: SUCCESS (in 118 ms)
  it_does_not_force_a_run_if_upstream_have_errors: SUCCESS (in 57 ms)
  it_sets_uptodate: SUCCESS (in 53 ms)
  it_calls_in_order: SUCCESS (in 67 ms)
  it_does_run_on_upstream_change_even_if_previous_success: SUCCESS (in 70 ms)
  it_runs_all_targets_when_dirty: SUCCESS (in 75 ms)
  it_calls_the_task_success: SUCCESS (in 46 ms)
  it_forces_a_run_even_if_not_dirty: SUCCESS (in 68 ms)
  """


def zest_Pipeline_calls():
    m_gc = zest.stack_mock(Pipeline._gc)
    m_p = zest.stack_mock(Pipeline._p)
    m_task1_start = zest.stack_mock(Task1.start)
    m_task2_start = zest.stack_mock(Task2.start)
    m_task3_start = zest.stack_mock(Task3.start)
    m_task1_is_dirty = zest.stack_mock(Task1.is_dirty)
    m_task2_is_dirty = zest.stack_mock(Task2.is_dirty)
    m_task3_is_dirty = zest.stack_mock(Task3.is_dirty)
    src_dir, dst_dir, tasks = None, None, None

    def _before():
        nonlocal src_dir, dst_dir, tasks
        src_dir = local.path("/tmp/test_pipeline")
        dst_dir = src_dir / "output"
        src_dir.delete()
        dst_dir.delete()
        src_dir.mkdir()

        tasks = dict(
            # (cls, params, kwargs)
            task1=(Task1, dict(inputs=dict(src_dir="")), {}),
            task2=(Task2, dict(inputs=dict(task1="../task1")), {}),
            task3=(Task3, dict(inputs=dict(task1="../task1", task2="../task2")), {}),
        )

    def _after():
        src_dir.delete()
        dst_dir.delete()

    def _set_dirty(dirty1, dirty2, dirty3):
        m_task1_is_dirty.returns(dirty1)
        m_task2_is_dirty.returns(dirty2)
        m_task3_is_dirty.returns(dirty3)

    def _p(**kwargs):
        return Pipeline(src_dir, dst_dir, tasks, **kwargs)

    def it_creates_dst_dirs():
        _set_dirty(True, True, True)
        _p()
        assert dst_dir.exists()
        assert (dst_dir / "task1").exists()
        assert (dst_dir / "task2").exists()
        assert (dst_dir / "task3").exists()

    def it_runs_all_targets_when_dirty():
        _set_dirty(True, True, True)
        _p()
        assert m_task1_start.called_once()
        assert m_task2_start.called_once()
        assert m_task3_start.called_once()

    def it_cleans_output_folders():
        _set_dirty(True, True, True)
        stuff = dst_dir / "task1"
        stuff.mkdir()
        stuff /= "stuff"
        stuff.touch()
        _p(clean=True)
        assert not stuff.exists()

    def it_cleans_and_early_outs():
        _set_dirty(True, True, True)
        _p(clean=True)
        assert not m_task1_start.called()
        assert not m_task2_start.called()
        assert not m_task3_start.called()

    def it_calls_in_order():
        time1 = None

        def _start_task1(*args, **kwargs):
            nonlocal time1
            time1 = time.time()

        time2 = None

        def _start_task2(*args, **kwargs):
            nonlocal time2
            time2 = time.time()

        time3 = None

        def _start_task3(*args, **kwargs):
            nonlocal time3
            time3 = time.time()

        m_task1_start.hook(_start_task1)
        m_task2_start.hook(_start_task2)
        m_task3_start.hook(_start_task3)
        _set_dirty(True, True, True)
        _p()
        assert time1 < time2 and time1 < time3 and time2 < time3
        m_task1_start.hook(None)
        m_task2_start.hook(None)
        m_task3_start.hook(None)

    def it_skips_task1_when_already_done():
        _set_dirty(False, True, False)
        _p()
        assert not m_task1_start.called()
        assert m_task2_start.called()
        assert not m_task3_start.called()
        # Note that task2 doesn't actually dirty anything so task 3 will not run
        # since the mock on set_dirty is set to False

    def it_runs_only_on_limited_targets():
        _set_dirty(True, True, True)
        _p(limit=["task2"])
        assert not m_task1_start.called()
        assert m_task2_start.called_once()
        assert not m_task3_start.called()

    def it_traps_exceptions_in_tasks():
        with zest.mock(Task1.error) as m_error:
            _set_dirty(True, False, False)
            e = Exception("problem")
            m_task1_start.exceptions(e)
            _p()
            assert m_error.called_once_with_kws(e=e)
            m_task1_start.exceptions(None)

    def it_does_not_run_if_upstream_error():
        _set_dirty(True, True, True)
        e = Exception("problem")
        m_task1_start.exceptions(e)
        _p()
        assert m_task1_start.called_once()
        assert not m_task2_start.called()
        assert not m_task3_start.called()
        assert (dst_dir / "task1" / PipelineState.error).exists()
        m_task1_start.exceptions(None)

    def it_forces_a_run_even_if_not_dirty():
        _set_dirty(False, False, False)
        _p(force=True)
        assert m_task1_start.called_once()
        assert m_task2_start.called_once()
        assert m_task3_start.called_once()

    def it_does_not_force_a_run_if_ignored():
        _set_dirty(True, False, False)
        _p(limit=["task2"], force=True)
        assert not m_task1_start.called()
        assert m_task2_start.called_once()
        assert not m_task3_start.called()

    def it_does_not_force_a_run_if_upstream_have_errors():
        _set_dirty(True, True, True)
        e = Exception("problem")
        m_task1_start.exceptions(e)
        _p(force=True)
        assert m_task1_start.called_once()
        assert not m_task2_start.called()
        assert not m_task3_start.called()
        m_task1_start.exceptions(None)

    def it_reruns_on_a_previous_failure():
        with zest.mock(
            Task1.get_output_state, returns=PipelineState.error
        ) as m_get_output_state:
            _set_dirty(True, False, False)
            _p()
            assert m_task1_start.called_once()
            assert not m_task2_start.called()
            assert not m_task3_start.called()

    def it_does_not_rerun_a_previous_success_if_no_upstream_changes():
        _set_dirty(False, False, False)
        _p()
        assert not m_task1_start.called()
        assert not m_task2_start.called()
        assert not m_task3_start.called()

    def it_does_run_on_upstream_change_even_if_previous_success():
        with zest.mock(
            Task1.get_output_state, returns=PipelineState.success
        ) as m_get_output_state:
            _set_dirty(True, True, True)
            Pipeline(src_dir, dst_dir, tasks)
            assert m_task1_start.called_once()
            assert m_task2_start.called_once()
            assert m_task3_start.called_once()

    def it_does_rerun_on_a_previous_failure():
        task1 = dst_dir / "task1"
        task1.mkdir()
        (task1 / PipelineState.error).touch()
        _set_dirty(True, True, True)
        Pipeline(src_dir, dst_dir, tasks)
        assert m_task1_start.called_once()
        assert m_task2_start.called_once()
        assert m_task3_start.called_once()

    def it_calls_the_task_success():
        with zest.mock(Task1.success) as m_success:
            _set_dirty(True, False, False)
            _p()
            assert m_success.called_once()

    def it_sets_uptodate():
        _set_dirty(False, False, False)
        _p()
        for task in ("task1", "task2", "task3"):
            assert (dst_dir / task / PipelineState.uptodate).exists()

    def it_clears_old_state():
        _set_dirty(True, False, False)
        e = Exception("problem")
        m_task1_start.exceptions(e)
        _p()
        assert m_task1_start.called_once()
        assert (dst_dir / "task1" / PipelineState.error).exists()
        m_task1_start.exceptions(None)

        # Let the second call pass
        m_task1_start.reset()
        _p()
        assert not (dst_dir / "task1" / PipelineState.error).exists()
        assert (dst_dir / "task1" / PipelineState.success).exists()

    def it_logs_state_changes():
        _set_dirty(True, False, False)
        p = _p()
        assert len(p._logs) == 4
        assert p._logs[0][1:3] == ("task1", "started")
        assert p._logs[1][1:3] == ("task1", "success")
        assert p._logs[2][1:3] == ("task2", "uptodate")
        assert p._logs[3][1:3] == ("task3", "uptodate")
        assert p.failed_count() == 0

    def it_logs_errors():
        _set_dirty(True, False, False)
        e = ValueError("test problem")
        m_task1_start.exceptions(e)
        p = _p()
        assert len(p._logs) == 4
        assert p._logs[0][1:3] == ("task1", "started")
        assert p._logs[1][1:4] == ("task1", "failed", "ValueError: test problem")
        assert p.failed_count() == 1
        m_task1_start.exceptions(None)

    zest()
