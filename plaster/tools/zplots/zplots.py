"""
A wrapper around Bokeh with the following goals:

    * Each plot is a one-liner.
        No need to call figure, add glyphs, renders, and then save
    * Plots can be easily merged or tiled
    * Pass fiddly-arguments to the underlying renders with f_ (for figures) or to the glyph without f_
    * Set up DataSources automatically if not provided so that hovering always works
    * Create thumbnails as HTML if requested
    * Simplified palette and color selection

Examples:
    # No need to call figure() or save()
    z.scat(x=[1, 2, 3], y=[4, 5, 6], f_title="A scatter plot")

    # This will produce 2 side by side merged scatter plots
    with z(_cols=2):
        with z(_merge=True, f_title="Some scatters"):
            z.scat(x=[1, 2, 3], y=[4, 5, 6])
            z.scat(x=[5, 2, 4], y=[1, 5, 4])
        with z(_merge=True, f_title="Some scatters"):
            z.scat(x=[1, 2, 3], y=[4, 5, 6])
            z.scat(x=[5, 2, 4], y=[1, 5, 4])

Setup for a notebook:
    from plaster.tools.zplots import zplots
    z = zplots.setup()
    z.scat(...)
    with z(...):

TO DO:
    * im stack
    * _x_ticks, _y_ticks: customize axis tick labels
        - tfb added very basic functionality allowing you to pass bokeh-recognized kw:
            _x_ticks = Munch(precision=0, use_scientific=True, power_limit_high=0, power_limit_low=0)
            kwargs['_x_ticks'] = _x_ticks
    * _thumbnail: Save to an html ALSO
    * Stats: Show mean, 1 std, and min-max
    * I'd rather have scat be able to take the args as well as kwargs, need a better pattern.
"""

import copy
import warnings
from itertools import cycle

import numpy as np
from munch import Munch
from plaster.tools.image.coord import HW, ROI, WH, XY, YX
from plaster.tools.log.log import debug
from plaster.tools.utils import utils
from plaster.tools.utils.data import arg_subsample, cluster, subsample
from plaster.tools.schema import check


def trap():
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception:
                args[0].reset()
                raise

        return wrapper

    return real_decorator


class ZPlots:
    feature = "dodgerblue"
    compare1 = "darkgray"
    compare2 = "lightgray"

    def __init__(self):
        """This is usually constructed by using the global setup() below"""
        self.reset()

    def reset(self):
        self.stack = [Munch()]
        self.figs = []  # Used for grids

    def __call__(self, **kws):
        """
        Used for "with" contexts like:
            with z(_prop=val):
        """
        self.stack.append(Munch(**kws))
        return self

    def __enter__(self):
        """
        If entering a context with "_cols" then a grid is created
        If entering with a "_merge" then we're in "merge" state
        Resets color cycling.
        """
        cols = self.stack[-1].get("_cols")

        this_merge = self.stack[-1].get("_merge")
        stack_merge = self._u_stack(exclude_last=True).get("_merge")
        if this_merge is not None and stack_merge is None:
            # If we are starting a merge and there is no merge in the stack above this
            assert cols is None
            self.figs += [self._figure()]
        elif cols is not None:
            self.stack[-1]["__grid"] = True
            assert self._u_stack().get("_merge", False) is False
            assert len(self.figs) == 0

        # A context will reset the color cycling unless it is in the stack already
        ustack = self._u_stack()
        if ustack.get("__colors") is None:
            from bokeh.palettes import Category20

            self.stack[-1]["__colors"] = cycle(Category20[20])

            # I want to be able to call "z.next()" and get the next color
            # after I use the default color. The first color in the palette
            # *is* the default color so I need to advance the iterator by 1
            # so that I will get the next non-default on the first call.
            next(self.stack[-1]["__colors"])

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Upon exit, show if a this is the end of a stand-alone merge context or
        if is the end of a __grid context.
        """

        if exc_type is not None:
            self.reset()
            return

        try:
            n_figs = len(self.figs)
            stack_merge = self._u_stack(exclude_last=True).get("_merge")
            if self.stack[-1].get("_merge") is not None and stack_merge is None:
                # If we closing a merge and there is no merge above this
                if self._u_stack().get("__grid") is None:
                    # If there's grid in the stack then need to show this merge
                    assert n_figs == 1
                    self._show(self.figs[0])
                    self.figs = []

            if self.stack[-1].get("__grid") is not None:
                # Closing a grid
                cols = self.stack[-1].get("_cols")
                if cols is not None and n_figs > 0:
                    rows = (n_figs // cols) + (1 if n_figs % cols else 0)
                    if n_figs < rows * cols:
                        self.figs += [None] * (rows * cols - n_figs)

                    arr = np.array(self.figs).reshape(rows, cols)

                    from bokeh.layouts import gridplot  # Defer slow imports

                    extra_gridplot_kwargs = {}
                    if self._u_stack().get("_notools"):
                        extra_gridplot_kwargs["toolbar_location"] = None

                    self._show(gridplot(list(arr), **extra_gridplot_kwargs))
                    self.figs = []

            self.stack.pop()
        except Exception as e:
            self.reset()
            raise e

    def _add_legend(self, fig):
        from bokeh.plotting import Figure  # Defer slow imports

        ustack = self._u_stack()
        legend = ustack.get("_legend")
        if isinstance(fig, Figure) and legend is not None:
            fig.legend.visible = True
            fig.legend.location = legend if legend is not True else "top_right"
            fig.legend.click_policy = "hide"

    def _figure(self):
        """
        Create a figure with f_* params and defaults; do not add to self._figs
        (Adding to _figs is the responsibility of the caller)
        """
        from bokeh.plotting import Figure  # Defer slow imports

        f_props = self._f_stack()

        # Cause save tool to save images as .svg instead of default .png
        # Note that this will likely cause problems with some plots that
        # make use of glyphs that don't work well in SVG.  My guess is that
        # this has been fixed in a more recent release of Bokeh and perhaps
        # we should bump the version we're using. tfb 13 April 2020
        # _f_defaults = dict(plot_width=350, plot_height=350, output_backend='svg')
        # This indeed causes scatter plots with circle glyphs to be empty,
        # so removing until we figure this out.  Users who need svg can pass
        # f_output_backend='svg' to the plotting routine to cause the tools
        # export icon to export an svg instead of default png.

        _f_defaults = dict(plot_width=350, plot_height=350)

        for key, val in _f_defaults.items():
            if key not in f_props:
                f_props[key] = val

        return Figure(**f_props)

    def _show(self, fig):
        """
        Show the fig (or grid)
        """
        from bokeh.io import show  # Defer slow imports

        self._add_legend(fig)

        show(fig)

    def _merge_stack(self, filter, transform, exclude_last=False):
        """
        Create a prop context from the stack so that the last
        element on the stack takes precedence.
        """
        m = {}
        for s in self.stack[0 : (-1 if exclude_last else None)]:
            for prop, val in s.items():
                if filter(prop):
                    m[transform(prop)] = val
        return m

    def _pre_get(self, kws, key, default_val=None):
        """
        Sometimes we need to a value from key before we can call _begin.
        Typically the _begin call configures the stack but in these
        special cases we want to check a value and not change the stack.
        """
        if key in kws:
            return kws[key]
        return self._merge_stack(
            filter=lambda prop: prop == key, transform=lambda prop: prop,
        ).get(key, default_val)

    def _u_stack(self, exclude_last=False):
        """
        All that start with underscore
        """
        return self._merge_stack(
            filter=lambda prop: prop.startswith("_"),
            transform=lambda prop: prop,
            exclude_last=exclude_last,
        )

    def _p_stack(self):
        """
        All non "f_" and non "_"
        """
        return self._merge_stack(
            filter=lambda prop: not prop.startswith("f_") and not prop.startswith("_"),
            transform=lambda prop: prop,
        )

    def _f_stack(self):
        """
        All f_* properties
        """
        return self._merge_stack(
            filter=lambda prop: prop.startswith("f_"), transform=lambda prop: prop[2:],
        )

    def _add_prop(self, key, val):
        self.stack[-1][key] = val

    def _apply_fig_props(self, fig):
        ustack = self._u_stack()
        if ustack.get("_size"):
            ustack["_size_x"] = ustack.get("_size")
            ustack["_size_y"] = ustack.get("_size")

        if ustack.get("_size_x"):
            fig.plot_width = ustack.get("_size_x")

        if ustack.get("_size_y"):
            fig.plot_height = ustack.get("_size_y")

        if ustack.get("_noaxes"):
            ustack["_noaxes_x"] = True
            ustack["_noaxes_y"] = True

        if ustack.get("_noaxes_x"):
            fig.xaxis.major_tick_line_color = None
            fig.xaxis.minor_tick_line_color = None
            fig.xaxis.major_label_text_font_size = "0pt"

        if ustack.get("_noaxes_y"):
            fig.yaxis.major_tick_line_color = None
            fig.yaxis.minor_tick_line_color = None
            fig.yaxis.major_label_text_font_size = "0pt"

        if ustack.get("_nogrid"):
            ustack["_nogrid_x"] = True
            ustack["_nogrid_y"] = True

        if ustack.get("_nogrid_x"):
            fig.xgrid.grid_line_color = None

        if ustack.get("_nogrid_y"):
            fig.ygrid.grid_line_color = None

        # ticks functionality is currently limited to passing bokeh-recognized keywords
        # to BasicTickFormatter.
        if ustack.get("_x_ticks"):
            from bokeh.models import BasicTickFormatter  # Defer slow imports

            fig.xaxis[0].formatter = BasicTickFormatter(**ustack.get("_x_ticks"))
        if ustack.get("_y_ticks"):
            from bokeh.models import BasicTickFormatter  # Defer slow imports

            fig.yaxis[0].formatter = BasicTickFormatter(**ustack.get("_y_ticks"))

        if ustack.get("_x_range_padding"):
            fig.x_range.range_padding = ustack.get("_x_range_padding")
        if ustack.get("_y_range_padding"):
            fig.y_range.range_padding = ustack.get("_y_range_padding")

        if ustack.get("_x_axis_label_orientation"):
            fig.xaxis.major_label_orientation = ustack.get("_x_axis_label_orientation")
        if ustack.get("_y_axis_label_orientation"):
            fig.yaxis.major_label_orientation = ustack.get("_y_axis_label_orientation")

        if ustack.get("_notools"):
            fig.tools = []
            fig.toolbar.logo = None
            fig.toolbar_location = None

        _range = ustack.get("_range")
        if _range is not None:
            assert len(_range) == 4
            ustack["_range_x"] = (_range[0], _range[1])
            ustack["_range_y"] = (_range[2], _range[3])

        _pad = ustack.get("_pad", 0.05)

        from bokeh.models.ranges import Range1d  # Defer slow imports

        _range_x = ustack.get("_range_x")
        if _range_x is not None:
            x_pad = (_range_x[1] - _range_x[0]) * _pad
            fig.x_range = Range1d(_range_x[0] - x_pad, _range_x[1] + x_pad)

        _range_y = ustack.get("_range_y")
        if _range_y is not None:
            y_pad = (_range_y[1] - _range_y[0]) * _pad
            fig.y_range = Range1d(_range_y[0] - y_pad, _range_y[1] + y_pad)

    def _build_column_data_source(self, kws, source_defaults):
        """
        Builds a Column Data Source from the kws and source_defaults
        """
        n_rows = None
        source_kwargs = {}
        ustack = self._u_stack()
        allow_labels = ustack.get("_no_labels") is not True

        for key, def_val in source_defaults.items():
            # Fill in the source using the order: ustack, if not then kws, if not then source_default
            val = ustack.get(key)
            if val is None:
                val = kws.get(key)
            if val is None:
                val = def_val

            if key != "_label":
                # Store the n_rows because this is needed to make the default labels
                # and sanity check that it is consistent
                assert n_rows is None or len(val) == n_rows
                n_rows = len(val)

            source_kwargs[key] = val

        # Extend _label to full length if needed
        if source_kwargs.get("_label") is not None:
            if not isinstance(source_kwargs.get("_label"), (list, np.ndarray, tuple)):
                source_kwargs["_label"] = [source_kwargs.get("_label")] * n_rows

        # REMOVE each of those source_defaults form the kws if present
        for key in source_defaults.keys():
            if key in kws:
                del kws[key]

        from bokeh.plotting import ColumnDataSource

        if not allow_labels:
            utils.safe_del(source_kwargs, "_label")
        else:
            kws["_label"] = "_label"

        kws["source"] = ColumnDataSource(source_kwargs)
        return kws

    def _begin(self, kws, source_defaults, **defaults):
        """
        Create a DataSource if it wasn't passed in.

        Add defaults.

        If in merge context then use the last fig.
        If not, create a new figure.

        Arguments:
            kws: The incoming kws
            source_defaults: A dict of the source fields and their defaults
                If the key of the dict is in the kws then that passed-in
                value is used, otherwise the value of this dict is
                placed into the source.
            defaults:
                Adds these defaults to the kws if they are not already present
                (Do not confuse these property defaults with the source_defaults)
        """
        from bokeh.plotting import ColumnDataSource

        pstack = self._p_stack()

        source = pstack.get("source")

        if kws.get("source") is not None:
            # There's a local source given, make a ColumnDataSource out of it
            source = kws["source"] = ColumnDataSource(kws.get("source"))

        # BUILD a column data source if needed
        n_source_fields = len(source_defaults)
        if source is None and n_source_fields > 0:
            kws = self._build_column_data_source(kws, source_defaults)
            source = kws["source"]

        # ADD requested defaults to kws if not already present in kws
        for key, val in defaults.items():
            if key not in kws:
                kws[key] = val

        self.stack.append(Munch(**kws))

        if self._u_stack().get("_merge"):
            fig = self.figs[-1]
        else:
            fig = self._figure()
            self.figs += [fig]

        from bokeh.models import HoverTool

        # In the case that a source is given then the _label refers to the column name
        # which can be pulled out of the stack (this has to happen after the above stack append)
        ustack = self._u_stack()
        label_col_name = ustack.get("_label")
        allow_labels = ustack.get("_no_labels") is not True

        # WIP: This is a start at generic extra tooltips
        # debug(source.to_df())
        # extra_hover_cols = []
        # _hover = ustack.get("_hover")
        # if _hover is not None:
        #     check.t(_hover, dict)
        #     for key, val in _hover.items():
        #         extra_hover_cols += [(key, "$" + key + "{0,0.0}")]
        #         source.add(val, key)

        if allow_labels:
            label_col = (
                []
                if label_col_name not in kws["source"].to_df().columns
                else [("label", f"@{label_col_name}")]
            )
            fig.add_tools(
                HoverTool(
                    tooltips=label_col
                    + [
                        ("x", "$x{0,0.0}"),
                        ("y", "$y{0,0.0}"),
                        ("value", "@image{0,0.0}"),
                    ],
                )
            )
        fig.toolbar.logo = None

        if ustack.get("_legend") is not None and allow_labels:
            # TFB NOTE: I'm not sure the intent of this logic.  I was providing a custom _label
            # in kwargs to a high-level plot routine, which was working, and when I tried to
            # move the legend with _legend='bottom_right' it caused my label to not show up, and
            # instead get replaced with some default based on label_col_name.  Is _legend being
            # too heavily overloaded here?  I just wanted to change location information, not
            # labeling strategy.
            if (
                ustack.get("_legend") is True
            ):  # so I'll add this for "nonspecific" use of legend...
                self._add_prop(
                    "legend_label", str(kws["source"].to_df()[label_col_name].iloc[0])
                )

        self._apply_fig_props(fig)

        return fig

    def _end(self):
        """
        When a figure is done, if it is standalone (not grid and not merge)
        then it needs to be shown.
        """
        ustack = self._u_stack()
        if ustack.get("__grid") is None and ustack.get("_merge") is None:
            # This is a stand-alone call (not a grid, not a merge), show it.
            assert len(self.figs) == 1
            self._show(self.figs[0])
            self.figs = []

        if len(self.figs) > 0:
            self._add_legend(self.figs[-1])

        self.stack.pop()

    def _cspan(self, _im, **kws):
        ustack = self._u_stack()
        ustack.update(kws)

        _cper = ustack.get("_cper")
        if _cper is not None:
            # color by percentile
            assert isinstance(_cper, tuple) and len(_cper) == 2
            low, high = np.nanpercentile(_im, _cper)
        else:
            _cspan = ustack.get("_cspan")
            if _cspan is None:
                low = 0.0
                if _im.shape[0] == 0:
                    high = 1.0
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            high = np.nanmean(_im) + np.nanstd(_im) * 4
                        except Exception:
                            high = 1.0
            else:
                if isinstance(_cspan, (list, tuple)):
                    assert len(_cspan) == 2
                    low = _cspan[0]
                    high = _cspan[1]
                elif isinstance(_cspan, np.ndarray):
                    assert _cspan.shape == (2,)
                    low = _cspan[0]
                    high = _cspan[1]
                else:
                    assert isinstance(_cspan, (int, float))
                    low = 0
                    high = _cspan

        return low, high

    def _im_setup(self, im):
        from bokeh.models import LinearColorMapper  # Defer slow imports

        _im = im
        assert _im.ndim == 2

        ustack = self._u_stack()

        y = 0
        if ustack.get("_flip_y"):
            assert "f_y_range" not in ustack
            _im = np.flip(_im, axis=0)
            y = _im.shape[0]

        dim = HW(_im.shape)

        if dim.w == 0 or dim.h == 0:
            # Deal with zero dims gracefully
            dim = HW(max(1, dim.h), max(1, dim.w))
            _im = np.zeros(dim)

        _nan = ustack.get("_nan")
        if _nan is not None:
            _im = np.nan_to_num(_im)

        low, high = self._cspan(_im)

        pal = ustack.get("_palette", "viridis")
        if pal == "viridis":
            from bokeh.palettes import viridis

            pal = viridis(256)

        elif pal in ("gray", "grey",):
            from bokeh.palettes import gray

            pal = gray(256)

        elif pal == "inferno":
            from bokeh.palettes import inferno

            pal = inferno(256)

        # zero_color = ustack.get("_zero_color")
        # if zero_color is not None:
        #     pal = copy.copy(pal)
        #     pal[0] = zero_color

        cmap = LinearColorMapper(palette=pal, low=low, high=high)

        return dim, cmap, _im, y

    def _im_post_setup(self, fig, dim):
        ustack = self._u_stack()
        full_w, full_h = (ustack.get("_full_w", False), ustack.get("_full_h", False))

        if ustack.get("_full"):
            full_w, full_h = (True, True)

        if full_h:
            full_h = dim.h + 40  # TASK: This value is weird, need a way to derive it
            if ustack.get("_min_h") is not None:
                full_h = max(full_h, ustack.get("_min_h"))

        if full_w:
            full_w = dim.w + 20  # TASK: This value is weird, need a way to derive it
            if ustack.get("_min_w") is not None:
                full_w = max(full_w, ustack.get("_min_w"))

        if full_w:
            fig.plot_width = full_w

        if full_h:
            fig.plot_height = full_h

    def color_reset(self):
        ustack = self._u_stack()
        for s in self.stack[::-1]:
            if s.get("__colors") is not None:
                from bokeh.palettes import Category20

                s["__colors"] = cycle(Category20[20])

    def next(self):
        """Return the next color in the palette with cycling"""
        colors = self._u_stack().get("__colors")
        return next(colors)

    @trap()
    def scat(self, **kws):
        """
        Scatter. Adds labels of the range if not provided.
        """
        ustack = self._u_stack()
        _n_samples = ustack.get("_n_samples", kws.get("_n_samples"))
        if _n_samples is not None:
            samp_iz = arg_subsample(kws["x"], _n_samples)
            kws["x"] = kws["x"][samp_iz]
            kws["y"] = kws["y"][samp_iz]
        fig = self._begin(
            kws,
            dict(x=None, y=None, _label=np.arange(len(kws.get("x", [])))),
            line_color=None,
            x="x",
            y="y",
        )
        fig.scatter(**self._p_stack())
        self._end()

    @trap()
    def cols(self, top=None, **kws):
        """
        A column plot.
        """

        # Note the pattern: When an arg is allowed (in this case top)
        # then it must be added to the kws also for normal processing.
        kws["top"] = top

        ustack = self._u_stack()
        _x = ustack.get("_x", kws.get("_x"))
        if _x is None:
            _x = np.arange(len(top))

        fig = self._begin(
            kws,
            dict(x=_x, top=top, _label=_x),
            x="x",
            width=0.9,
            top="top",
            line_color=None,
        )

        fig.vbar(**self._p_stack())
        self._end()

    @trap()
    def hist(
        self, data=None, _vertical=None, **kws,
    ):
        """
        Histogram. Converts nan in data to zeros.

        _bins: If a 3-tuple it is converted to a linspace. Default=(min, max, 50)
        _density: Passed as "density" to np.histogram()
        _normalizer: Value to divide through by safely. Default=1.0)
        _step: If True draws steps lines instead of filled bars
        _subsample: If non None it will subsample this number of items from the set
        """

        if isinstance(data, str):
            assert kws.get("source") is not None
            data = kws.pop("source")[data]

        else:
            data = data.copy()

        ustack = self._u_stack()
        ustack.update(kws)
        _bins = ustack.get("_bins")
        _density = ustack.get("_density", False)
        _normalizer = ustack.get("_normalizer", 1.0)
        _step = ustack.get("_step", False)
        _subsample = ustack.get("_subsample")
        _remove_nan = ustack.get("_remove_nan")

        if _remove_nan:
            data = data[~np.isnan(data)]
        else:
            data = np.nan_to_num(data)

        if _subsample is not None:
            data = subsample(data, _subsample)

        if _bins is None:
            if data.shape[0] > 0:
                min_, max_ = np.percentile(data, (0, 99))
            else:
                min_ = 0
                max_ = 0
            _bins = np.linspace(min_, max_, 100)
        elif isinstance(_bins, (tuple, list)):
            _bins = np.linspace(_bins[0], _bins[1], _bins[2] if len(_bins) == 3 else 50)

        _hist, _edges = np.histogram(data, bins=_bins, density=_density)
        _hist = _hist.astype(float)
        _hist = utils.np_safe_divide(_hist, np.array(_normalizer), default=0.0)

        fig = self._begin(
            kws,
            dict(
                left=_edges[:-1],
                right=_edges[1:],
                top=_hist,
                bottom=np.zeros_like(_hist),
                _label="",
            ),
        )

        if _step:
            # Note, hovers do not work for steps as of Boken 1.4.0
            # https://github.com/bokeh/bokeh/issues/7419
            fig.step(x="left", y="top", mode="center", **self._p_stack())
        else:
            fig.quad(
                top="top",
                bottom="bottom",
                left="left",
                right="right",
                line_color=None,
                **self._p_stack(),
            )

        if _vertical is not None:
            fig.line(x=(_vertical, _vertical), y=(0, np.max(_hist)), color="red")

        self._end()

    @trap()
    def line(self, **kws):
        """
        Line.
        _step: If True they will be step function
        """
        def_range = np.arange(len(kws.get("y", [])))
        fig = self._begin(
            kws, dict(x=def_range, y=None, _label=def_range), x="x", y="y",
        )

        ustack = self._u_stack()
        pstack = self._p_stack()

        source = pstack.get("source")
        if "x" not in source.column_names:
            n_rows = len(source.to_df())
            kws["source"].add(np.arange(n_rows), "x")

        if ustack.get("_step"):
            fig.step(**pstack)
        else:
            fig.line(**pstack)

        _dots = ustack.get("_dots")
        if _dots:
            from bokeh.plotting import ColumnDataSource

            step = 1
            if isinstance(_dots, int):
                step = _dots

            df = pstack["source"].to_df().iloc[::step, :]
            pstack["source"] = ColumnDataSource(df)

            if pstack.get("line_color") is not None:
                pstack["fill_color"] = pstack.get("line_color")
            # if pstack.get("line_alpha") is not None:
            #     pstack["fill_alpha"] = pstack.get("line_alpha")

            fig.scatter(**pstack)

        self._end()

    @trap()
    def distr(self, samples, _percentiles=(0, 25, 50, 75, 100), _vertical=None, **kws):
        """
        A distribution plot shows percentiles 0, 25, 50, 75, 100 as whiskers
        """
        check.array_t(samples, ndim=2)

        kws["xs"] = []
        kws["ys"] = []
        kws["line_color"] = []
        kws["line_width"] = []
        for row_i, row in enumerate(samples):
            p0, p25, p50, p75, p100 = np.nanpercentile(row, _percentiles)

            kws["xs"] += [
                [p0, p100],  # Horizontal line
                [p25, p75],  # IQR horzontal
                [p50, p50],
            ]

            y = row_i
            kws["ys"] += [
                [y, y],  # Horizontal line
                [y, y],
                [y - 1.5, y + 1.5],
            ]

            kws["line_color"] += [
                "gray",
                "black",
                "white",
            ]

            kws["line_width"] += [
                1,
                2,
                2,
            ]

        fig = self._begin(
            kws,
            dict(xs=None, ys=None, line_color=None, line_width=None),
            xs="xs",
            ys="ys",
            line_color="line_color",
            line_width="line_width",
        )
        pstack = self._p_stack()
        fig.multi_line(**pstack)
        if _vertical is not None:
            fig.line(
                x=(_vertical, _vertical), y=(0, len(samples)), color="red"
            )
        self._end()

    @trap()
    def multi_line(self, **kws):

        if isinstance(kws.get("xs"), np.ndarray):
            kws["xs"] = kws["xs"].tolist()
        if isinstance(kws.get("ys"), np.ndarray):
            kws["ys"] = kws["ys"].tolist()

        def_range = np.arange(len(kws.get("ys", [])))
        fig = self._begin(
            kws, dict(xs=def_range, ys=None, _label=def_range), xs="xs", ys="ys",
        )

        ustack = self._u_stack()
        pstack = self._p_stack()

        source = pstack.get("source")
        if "xs" not in source.column_names:
            n_rows = len(source.to_df())
            kws["source"].add(np.arange(n_rows), "xs")

        # TASK: I just can't get this to work right
        # if ustack.get("_step"):
        #     from bokeh.plotting import ColumnDataSource
        #     df = pstack["source"].to_df()
        #     df = df.iloc[np.repeat(np.arange(len(df)), 2)]
        #     for i, _x in enumerate(df["xs"]):
        #         df["xs"][i] = _x[1::2] + _x[0::2] + 1
        #     pstack["source"] = ColumnDataSource(df)

        fig.multi_line(**pstack)

        self._end()

    @trap()
    def ellipse(self, **kws):
        fig = self._begin(
            kws,
            dict(x=None, y=None, width=None, height=None),
            x="x",
            y="y",
            width="width",
            height="height",
            fill_color=None,
        )
        fig.ellipse(**self._p_stack())
        self._end()

    @trap()
    def im(self, *args, **kwargs):
        """
        Short term hack to fix the issue where im munges the stack.

        The hack works by creating a new stack just for the im call that can get screwed up and will then be reset before we return.
        """
        with self():
            return self._im(*args, **kwargs)

    def _im(self, im_data=None, **kws):
        """
        Image.
        _cspan: Low and High heatmap range. Default=(0, mean + 4*std)
        _palette: "viridis", "inferno", "gray"
        _zero_color: If set, special color for zero
        _flip_y: If True flips y
        _full_w: Force figure to the width of the image
        _full_h: Force figure to the height of the image
        _full: Force figure to the width and height of the image
        _min_w: Minimum width
        _min_h: Minimum height
        _nan_color: What color to use to draw nan (bokeh named colors, etc.)
        """
        assert self._u_stack().get("source") is None

        # If I do this then I break the _cols checks. But I don't know
        # why it was I needed this...

        # It's because of cspan need in _im_setup but before _begin has done its magic
        # self.stack.append(Munch(**kws))
        self.stack[-1].update(kws)

        """
        TODO:
        There's a major problem here. The above code is updating the stack
        which means these props are not popped off the stack.
        The issue is that _im_setup needs props befopre it can do things
        so that it can pass those dims and other things into _begin
        But _begin is the one that is meant to be doing stack munging
        not here. So I need fix this circular dependency maybe a _begin_im
        that then calls _begin or similar?
        """

        ustack = self._u_stack()
        nan_color = ustack.get("_nan_color")

        _n_samples = ustack.get("_n_samples")
        if _n_samples is not None:
            _n_samples = ustack.get("_n_samples", 1000)
            im_data = subsample(im_data, _n_samples)

        is_nan_im = np.zeros_like(im_data, dtype=bool)
        if nan_color is not None:
            is_nan_im = np.isnan(im_data)
            im_data = np.where(is_nan_im, 0, im_data)

        dim, cmap, im_data, y = self._im_setup(im_data)

        extra_hover_cols = {}
        extra_hover_tooltips = []

        _hover_rows = ustack.get("_hover_rows")
        if _hover_rows is not None:
            check.t(_hover_rows, dict)
            for key, val in _hover_rows.items():
                val = np.array(val)
                val_im = np.full(im_data.shape, "", dtype=object)
                val_im[:] = val[:, None]
                extra_hover_cols[key] = [val_im]
                extra_hover_tooltips += [(key, "@" + key)]

        # See: "Image Hover" here https://docs.bokeh.org/en/latest/docs/user_guide/tools.html
        fig = self._begin(
            kws,
            dict(
                image=[im_data],
                x=kws.get("_x", [0]),
                y=kws.get("_y", [y]),
                dw=kws.get("_dim_w", [dim.w]),
                dh=kws.get("_dim_h", [dim.h]),
                **extra_hover_cols,
            ),
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
        )

        self._im_post_setup(fig, dim)

        if _hover_rows is not None:
            from bokeh.models import HoverTool

            fig.add_tools(
                HoverTool(
                    tooltips=[
                        ("x", "$x{0,0}"),
                        ("y", "$y{0,0}"),
                        ("value", "@image{0,0.0}"),
                    ]
                    + extra_hover_tooltips,
                )
            )

        fig.image(
            color_mapper=cmap, **self._p_stack(),
        )

        if nan_color is not None:
            color_im = np.zeros(dim, dtype=np.uint32)
            view = color_im.view(dtype=np.uint8).reshape((dim.h, dim.w, 4))

            from bokeh.colors import named

            rgb = getattr(named, nan_color.lower())
            view[:, :, 0] = rgb.r
            view[:, :, 1] = rgb.g
            view[:, :, 2] = rgb.b
            view[:, :, 3] = np.where(is_nan_im, 255.0, 0)
            fig.image_rgba(image=[color_im], x=[0], y=[y], dw=[dim.w], dh=[dim.h])

        self._end()

    @trap()
    def im_blend(self, im_data=None, alpha_im=None, **kws):
        """
        Blend image with an alpha map.
        """
        assert self._u_stack().get("source") is None
        dim, cmap, im_data, y = self._im_setup(im_data)

        n_colors = len(cmap.palette)
        rpal = np.array([int(p[1:3], 16) for p in cmap.palette])
        gpal = np.array([int(p[3:5], 16) for p in cmap.palette])
        bpal = np.array([int(p[5:7], 16) for p in cmap.palette])

        cmap_delta = cmap.high - cmap.low
        normalized_im = (im_data - cmap.low) / cmap_delta
        palette_scaled_im = (n_colors * normalized_im).astype(int).clip(min=0, max=255)

        color_im = np.zeros(dim, dtype=np.uint32)
        view = color_im.view(dtype=np.uint8).reshape((dim.h, dim.w, 4))

        view[:, :, 0] = rpal[palette_scaled_im]
        view[:, :, 1] = gpal[palette_scaled_im]
        view[:, :, 2] = bpal[palette_scaled_im]
        view[:, :, 3] = (255.0 * alpha_im).astype(int)

        # See: "Image Hover" here https://docs.bokeh.org/en/latest/docs/user_guide/tools.html
        fig = self._begin(
            kws,
            dict(
                image=[color_im],
                x=kws.get("_x", [0]),
                y=kws.get("_y", [y]),
                dw=kws.get("_dim_w", [dim.w]),
                dh=kws.get("_dim_h", [dim.h]),
            ),
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
        )

        self._im_post_setup(fig, dim)

        fig.image_rgba(**self._p_stack(),)
        self._end()
        return fig

    @trap()
    def im_color(self, gray=None, red=None, green=None, blue=None, **kws):
        """
        Color Image.
        Range defaults to 0.0-1.0
        See im() options
        """
        raise NotImplementedError
        # This code is not working at the moment

        assert self._u_stack().get("source") is None

        # Set primary image with a preference order
        channels = (gray, red, green, blue)
        pri_im = None
        for im in channels:
            if im is not None:
                pri_im = im
                break

        assert pri_im is not None

        dim, cmap, im_data, y = self._im_setup(pri_im)
        fig = self._begin(kws, dict())

        # ACCUMLATE into a float and then bound 0, 255
        f_color_im = np.zeros((dim.h, dim.w, 4))

        # For each color channel, allow _im_setup to run in case it needs to be inverted
        for channel, channel_iz in zip(channels, [(0, 1, 2), (0,), (1,), (2,)]):
            if channel is not None:
                _, _, im, _ = self._im_setup(channel, fig)
                for ch in channel_iz:
                    f_color_im[:, :, ch] = channel

        f_color_im *= 255.0

        np.clip(f_color_im, 0, 255, out=f_color_im)

        color_im = np.zeros(dim, dtype=np.uint32)
        view = color_im.view(dtype=np.uint8).reshape((dim.h, dim.w, 4))

        view[:] = f_color_im[:].astype(np.uint8)
        view[:, :, 3] = 255  # Alpha to 1.0

        fig.image_rgba(
            image=[color_im], x=[0], y=[y], dw=[dim.w], dh=[dim.h], **self._p_stack()
        )
        self._end()

    @trap()
    def im_signed(self, im_data=None, **kws):
        positive = np.clip(im_data, a_min=0, a_max=None)
        negative = np.clip(-im_data, a_min=0, a_max=None)
        self.im_color(red=negative, green=positive, **kws)

    @trap()
    def im_clus(self, data, _n_samples=None, **kws):
        _n_samples = self._pre_get(dict(_n_samples=_n_samples), "_n_samples", 500)
        _hover_rows = self._pre_get(kws, "_hover_rows")

        sample_row_iz = arg_subsample(data, _n_samples)
        im, order = cluster(data[sample_row_iz], return_order=True)

        if _hover_rows is not None:
            check.t(_hover_rows, dict)
            _dupe_hover_rows = {}
            for key, val in _hover_rows.items():
                val = np.array(val)
                assert val.shape[0] == data.shape[0]
                _dupe_hover_rows[key] = val[sample_row_iz][order]
            kws["_hover_rows"] = _dupe_hover_rows

        # f_title = kws.pop("f_title", "")
        # f_title += f" n_rows={data.shape[0]}"
        # self.im(im, f_title=f_title, **kws)
        self.im(im, **kws)

    @trap()
    def im_sort(self, data, **kws):
        _n_samples = self._pre_get(kws, "_n_samples", 1000)
        _hover_rows = self._pre_get(kws, "_hover_rows")
        row_iz = arg_subsample(data, _n_samples)
        row_iz = row_iz[np.argsort(np.nansum(data[row_iz], axis=1))]

        if _hover_rows is not None:
            check.t(_hover_rows, dict)
            _dupe_hover_rows = {}
            for key, val in _hover_rows.items():
                val = np.array(val)
                assert val.shape[0] == data.shape[0]
                _dupe_hover_rows[key] = val[row_iz]
            kws["_hover_rows"] = _dupe_hover_rows

        # f_title = kws.pop("f_title", "")
        # f_title += f" n_rows={data.shape[0]}"
        # self.im(data[row_iz], f_title=f_title, **kws)
        self.im(data[row_iz], **kws)

    @trap()
    def im_peaks(self, im, circle_im, index_im, sig_im, snr_im, asr_im, **kws):
        """
        This is a custom plot for drawing information about sigproc data

        Args:
            _circle_color (tuple) - Circle colors can be changed by passing _circle_color,
                                    which should be a tuple of the form:
                                    (R, G, B) or (R, G, B, A) where each value is 0-255
        """
        from bokeh.colors import named
        from bokeh.models import LinearColorMapper  # Defer slow imports
        from bokeh.models import HoverTool
        from bokeh.palettes import gray

        pal = gray(256)
        dim = HW(im.shape)

        low, high = self._cspan(im, **kws)
        cmap = LinearColorMapper(palette=pal, low=low, high=high)
        n_colors = len(cmap.palette)

        rpal = np.array([int(p[1:3], 16) for p in cmap.palette])
        gpal = np.array([int(p[3:5], 16) for p in cmap.palette])
        bpal = np.array([int(p[5:7], 16) for p in cmap.palette])

        cmap_delta = cmap.high - cmap.low
        normalized_im = (im - cmap.low) / cmap_delta
        palette_scaled_im = (n_colors * normalized_im).astype(int).clip(min=0, max=255)

        color_im = np.zeros(dim, dtype=np.uint32)
        view = color_im.view(dtype=np.uint8).reshape((dim.h, dim.w, 4))
        view[:, :, 0] = rpal[palette_scaled_im]
        view[:, :, 1] = gpal[palette_scaled_im]
        view[:, :, 2] = bpal[palette_scaled_im]
        view[:, :, 3] = 255

        ustack = self._u_stack()
        ustack.update(kws)
        circle_color = ustack.get("_circle_color", (0, 180, 255, 255))

        try:
            if len(circle_color) == 3:
                circle_color = (*circle_color, 255)

            if not all(v >= 0 and v <= 255 for v in circle_color):
                raise ValueError("_circle_color values must be between 0 and 255")
        except TypeError as e:
            raise TypeError(
                "_circle_color must be of form (R, G, B) or (R, G, B, A)"
            ) from e

        view[circle_im > 0, :] = circle_color

        # See: "Image Hover" here https://docs.bokeh.org/en/latest/docs/user_guide/tools.html
        self._add_prop("_no_labels", True)

        fig = self._begin(
            kws,
            dict(
                image=[color_im],
                x=[0],
                y=[0],
                dw=[dim.w],
                dh=[dim.h],
                peak_i=[index_im],
                sig=[sig_im],
                snr=[snr_im],
                asr=[asr_im],
                val=[im],
            ),
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
        )

        self._im_post_setup(fig, dim)

        fig.add_tools(
            HoverTool(
                tooltips=[
                    ("x", "$x"),
                    ("y", "$y"),
                    ("peak_i", "@peak_i"),
                    ("value", "@val{0,0.0}"),
                    ("sig", "@sig{0,0.0}"),
                    ("snr", "@snr{0,0.0}"),
                    ("asr", "@asr{0,0.0}"),
                ],
            )
        )

        fig.image_rgba(**self._p_stack())

        self._end()


def notebook_full_width():
    from IPython.core.display import HTML, display  # Defer slow imports

    display(HTML("<style>.container { width:100% !important; }</style>"))


def setup():
    from bokeh.io import output_notebook  # Defer slow imports

    np.set_printoptions(precision=1)
    output_notebook()
    notebook_full_width()
    return ZPlots()


'''
# TFB: I moved this over from the deprecated ipynb_helpers/zplots.py
# Do we want it intergrated into ZPlots, or moved into its own module?

# ZBS: 2020 Feb 20. This is deprecated repalced by z.feature, etc.

from bokeh.colors import HSL


class Pal:
    """
    This is designed so that it is easy to view for color blindness.

    "feature":
        Some element that you want to bring attention to.
        There should be only one or two such elements

    "compare":
        Element that are in the background.
        These can be at three levels: (0, 1, 2)

    Features and compares are all differentiated by brightness alone
    but you may assign them into a hue group so that non-color blind
    people can easily use the name of the colors while preserving
    the ability of color blind people to see it.

    Also, when there are multiple compares, it is good practice
    to use line_dash or other appropriate non-color modifiers.
    """

    hues = dict(
        red=0, orange=30, yellow=50, green=90, blue=210, violet=280, gray=0  # Special
    )

    sat = 1.0  # Set this to zero to see what a color blind person will see

    @classmethod
    def feature(cls, hue="green"):
        sat = 0 if hue == "gray" else cls.sat
        return HSL(cls.hues[hue], sat, 0.2).to_rgb()

    @classmethod
    def compare(cls, level=0, hue="blue"):
        assert 0 <= level < 3
        sat = 0 if hue == "gray" else cls.sat
        return HSL(cls.hues[hue], sat, 0.45 + level * 0.21).to_rgb()
'''
