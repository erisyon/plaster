import numpy as np
from functools import wraps
from plumbum import local
from plaster.tools.utils import utils
from munch import Munch
from plaster.tools.log.log import debug


def disk_memoize():
    def _wraps(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert isinstance(args[0], BaseResult)
            path = args[0]._folder / f"_cache_{func.__name__}.pkl"
            try:
                rv = utils.pickle_load(path)
            except Exception:
                rv = func(*args, **kwargs)
                utils.pickle_save(path, rv)
            return rv

        return wrapper

    return _wraps


class ArrayResult:
    """
    A common result type is a large array that is larger than memory.
    Instances of this class allow those to be written to files and then
    streamed as needed.
    """

    def __init__(self, filename, dtype, shape, mode="r", offset=0, order="C"):
        self._fp = None
        self.filename = filename
        self.dtype = dtype
        self.shape = shape
        self.mode = mode
        self.offset = offset
        self.order = order
        self.arr()

    def arr(self):
        if self._fp is None:
            self._fp = np.memmap(
                self.filename,
                dtype=self.dtype,
                shape=self.shape,
                mode=self.mode,
                offset=self.offset,
                order=self.order,
            )
        return self._fp

    def flush(self):
        if self._fp is not None:
            self._fp.flush()

    def reshape(self, new_shape):
        """Will truncate the size of the memory mapped file if getting smaller"""
        self.arr().base.resize(np.product(new_shape) * self.arr().itemsize)
        self.flush()
        del self._fp
        self._fp = None
        self.shape = new_shape
        if self.mode == "w+":
            # In the case that this was originally opened to overwrite
            # now that it has been truncated and will re-open on the
            # next call to arr() we must ensure that it doesn't overwrite
            # by changing the mode to "r+"
            self.mode = "r+"

    def __getstate__(self):
        self.flush()
        return (
            self.filename,
            self.dtype,
            self.shape,
            self.mode,
            self.offset,
            self.order,
        )

    def __setstate__(self, state):
        self._fp = None
        (
            self.filename,
            self.dtype,
            self.shape,
            self.mode,
            self.offset,
            self.order,
        ) = state
        # When re-loading, kick it in to read-only
        self.mode = "r"

    def __getitem__(self, index):
        fp = self.arr()
        return fp[index]

    def __setitem__(self, index, val):
        fp = self.arr()
        fp[index] = val


class BaseResult(Munch):
    """
    Base of all Results.

    This inherits from Munch and plays some games with overloading
    __getattr__ so that you can access fields with Just-In-Time loading.

    Normal result properties are stored in the munch are are
    saved into an indexed pickle.  However, _folder and _is_loaded_result
    are special properties that are handled differently.

    There's two ways to construct a Result
    1. Direct instanciation, typical for a worker.
    2. By loading from a directory (typical for a task or report)
       In this case load_from_folder() is used.
    """

    name = None  # Overload in subclass
    filename = None  # Overload in subclass

    def __init__(self, folder=None, is_loaded_result=False, **kwargs):
        if folder is None:
            folder = local.cwd

        # Use object __setattr__ to bypass Munch
        object.__setattr__(self, "_folder", local.path(folder))
        object.__setattr__(self, "_is_loaded_result", is_loaded_result)

        # Everything else will go into the munch
        super().__init__(**kwargs)

    def save(self):
        """
        The default behavior is to simply pickle the entire object.

        Some sub-classes might choose to defer to some non-pickle files
        for debugging information or other.

        kwargs are added into to the save if specified.
        """

        # Sanity check that required fields are present
        for prop, type_ in self.required_props.items():
            if prop not in self:
                raise ValueError(f"Required property '{prop}' not found")
            if not isinstance(self[prop], type_):
                raise ValueError(
                    f"Property '{prop}' of wrong type; expected '{type_}' found '{type(self[prop])}'"
                )

        # Note, self.filename is a class member set in the subclass
        utils.indexed_pickler_dump(self, self._folder / self.filename)

    def keys(self):
        """Get the key list from the pickle if this is a loaded result"""
        _is_loaded_result = object.__getattribute__(self, "_is_loaded_result")
        if _is_loaded_result:
            # Bypass munch for these elements...
            _folder = object.__getattribute__(self, "_folder")
            filename = object.__getattribute__(self, "filename")
            return list(utils.indexed_pickler_load_keys(_folder / filename)) + [
                "_folder"
            ]
        return super().keys()

    def __getstate__(self):
        """
        This is overloading the pickle save to ensure that the _folder
        property ends up in the pickle. _folder is NOT stored in the munch
        """
        d = Munch.__getstate__(self)
        d["_folder"] = str(object.__getattribute__(self, "_folder"))
        return d

    def __setstate__(self, state):
        """
        Extract _folder from the pickle.  See above.
        """
        super().__setstate__(state)
        object.__setattr__(self, "_folder", local.path(state["_folder"]))

    def __getattr__(self, key):
        """
        Called when the attribute is not found.

        This is tricky because this class inherits from Munch
        which also overloads __getattr__.

        We first ask the munch if it has the attr,
        if not then we attempt to load it from the indexed pickle
        and then as Munch to try again.
        """
        try:
            val = Munch.__getattr__(self, key)
            if isinstance(val, ArrayResult):
                return val.arr()
            return val
        except AttributeError:
            # Bypass munch for these elements...
            _folder = object.__getattribute__(self, "_folder")
            filename = object.__getattribute__(self, "filename")

            with local.cwd(_folder):
                loaded = utils.indexed_pickler_load(
                    filename, prop_list=[key], skip_missing_props=True,
                )

                if isinstance(loaded[key], ArrayResult):
                    setattr(self, key, loaded[key].arr())
                else:
                    setattr(self, key, loaded[key])

                # Try again
                try:
                    return Munch.__getattr__(self, key)
                except AttributeError:
                    # (An attempt to deal with attributes that get added to
                    # task Results but that can be allowed to be None -
                    # and gracefully allowing us to load versions of the
                    # Result klass in which this attribute is missing.)
                    #
                    # The attribute is also not in the pickle.  If this
                    # attribute is required and allowed to be None, set
                    # it so, else raise.
                    required_props = object.__getattribute__(self, "required_props")
                    types = required_props.get(key, None)
                    if isinstance(types, tuple) and type(None) in types:
                        self[key] = None
                        return None
                    # key not found or not allowed to be None:
                    raise AttributeError

    @classmethod
    def load_from_folder(cls, folder, prop_list=None):
        """
        Used when one task needs the result of a previous stage.
        Pass in prop_list to load only certain properties.

        The _folder property is treated specially, it is not
        part of the Munch
        """
        folder = local.path(folder)
        with local.cwd(folder):
            inst = cls(folder, is_loaded_result=True)
            loaded = utils.indexed_pickler_load(
                folder / cls.filename, prop_list=prop_list
            )

            for key in loaded:
                if isinstance(loaded[key], ArrayResult):
                    setattr(inst, key, loaded[key].arr())
                else:
                    setattr(inst, key, loaded[key])

            object.__setattr__(inst, "_folder", folder)

        return inst

    @classmethod
    def task_names(cls, run):
        """
        Sometimes a run will contain multiple instances of the same task, each
        with its own name.  In general, code dealing in results will not know
        these names ahead of time, so this fn provides a way to get a list of
        task names for a given task class so that the BaseResult-derived result
        can be loaded via the JIT mechanism, e.g.

        names = MyResultClass:task_names(run)
        my_class_results = [ run[name] for name in names ]
        """
        names = []
        for task_name, task_block in run.manifest.tasks.items():
            if task_name == cls.name or (
                "task" in task_block and task_block.task == cls.name
            ):
                names += [task_name]
        return names
