import itertools
import numpy as np
from munch import Munch

from plaster.tools.image import imops
from plaster.tools.image.coord import XY
from plaster.tools.ipynb_helpers import displays
from plaster.tools.log.log import debug
from plaster.tools.schema import check
from plaster.tools.utils import utils, data
from plaster.tools.zplots.zplots import ZPlots


# Mature
# -------------------------------------------------------------------------------------


def plot_psfs(psfs, scale=1.0, **kwargs):
    """
    Show regional PSF in a summary image.
    Arguments:
        psfs is 4-d array: (regional divs y, regional divs x, psf h, psf w)
    """
    divs_h, divs_w, dim_h, dim_w = psfs.shape
    assert divs_h == divs_w
    divs = divs_h
    assert dim_h == dim_w
    dim = dim_h

    z = kwargs.pop("_zplots_context", None) or ZPlots()
    with z(_size=kwargs.get("_size", max(100, int(dim * divs * scale)))):
        comp = np.zeros((divs * dim, divs * dim))
        for y_i, x_i in itertools.product(range(divs), range(divs)):
            comp[y_i * dim : (y_i + 1) * dim, x_i * dim : (x_i + 1) * dim] = psfs[
                y_i, x_i
            ]
        z.im(comp, **kwargs)


def _sigproc_v2_im(im, locs, sig=None, snr=None, keep_mask=None, **kwargs):
    # if keep_mask is not None:
    #     locs = locs[keep_mask]
    #     sig = sig[keep_mask]
    #     snr = snr[keep_mask]

    z = kwargs.pop("_zplots_context", None) or ZPlots()

    n_locs = len(locs)

    if sig is None:
        sig = np.full((n_locs,), np.nan)

    if snr is None:
        snr = np.full((n_locs,), np.nan)

    circle_im = circle_locs(
        im, locs, inner_radius=6, outer_radius=7, fill_mode="one", keep_mask=keep_mask
    )
    index_im = circle_locs(
        im, locs, inner_radius=0, outer_radius=7, fill_mode="index", keep_mask=keep_mask
    )
    sig_im = circle_locs(
        im,
        locs,
        inner_radius=0,
        outer_radius=7,
        fill_mode="vals",
        vals=sig,
        keep_mask=keep_mask,
    )
    snr_im = circle_locs(
        im,
        locs,
        inner_radius=0,
        outer_radius=7,
        fill_mode="vals",
        vals=snr,
        keep_mask=keep_mask,
    )

    z.im_peaks(im, circle_im, index_im, sig_im, snr_im, **kwargs)


def sigproc_v2_im(run, fl_i=0, ch_i=0, cy_i=0, keep_mask=None, **kwargs):
    """
    Show a sigproc_v2 static view with roll-over information
    """
    im = run.sigproc_v2.aln_ims[fl_i, ch_i, cy_i]
    locs = run.sigproc_v2.locs(fl_i)
    sig = run.sigproc_v2.sig(fl_i)[:, ch_i, cy_i]
    snr = run.sigproc_v2.snr(fl_i)[:, ch_i, cy_i]

    _sigproc_v2_im(im, locs, sig, snr, keep_mask=None, **kwargs)


def circle_locs(
    im, locs, inner_radius=3, outer_radius=4, fill_mode="nan", vals=None, keep_mask=None
):
    """
    Returns a copy of im with circles placed around the locs.

    Arguments
        im: The background image
        locs: Nx2 matrix of peak locations
        circle_radius: Radius of circle to draw
        fill_mode:
            "nan": Use im and overlay with circles of NaNs
            "index": zero for all background and the loc index otherwise
                     (This causes the loss of the 0-th peak)
            "one": zero on background one on foreground
            "vals": zero on background val[loc_i[ for foreground
        style_mode:
            "donut" Draw a 1 pixel donut
            "solid": Draw a filled circle

    Notes:
        This can then be visualized like:
            circle_im = circle_locs(im, locs, fill_mode="nan")
            z.im(circle_im, _nan_color="red")
    """
    n_locs = len(locs)
    if keep_mask is None:
        keep_mask = np.ones((n_locs,), dtype=bool)

    mea = (outer_radius + 1) * 2 + 1
    hat = imops.generate_circle_mask(inner_radius, mea)
    brim = imops.generate_circle_mask(outer_radius, mea)
    brim = brim & ~hat

    if fill_mode == "nan":
        circle_im = np.zeros_like(im)
        for loc_i, (loc, keep) in enumerate(zip(locs, keep_mask)):
            if keep:
                imops.set_with_mask_in_place(circle_im, brim, 1, loc=loc, center=True)
        return np.where(circle_im == 1, np.nan, im)

    if fill_mode == "index":
        circle_im = np.zeros_like(im)
        for loc_i, (loc, keep) in enumerate(zip(locs, keep_mask)):
            if keep:
                imops.set_with_mask_in_place(
                    circle_im, brim, loc_i, loc=loc, center=True
                )
        return circle_im

    if fill_mode == "one":
        circle_im = np.zeros_like(im)
        for loc_i, (loc, keep) in enumerate(zip(locs, keep_mask)):
            if keep:
                imops.set_with_mask_in_place(circle_im, brim, 1, loc=loc, center=True)
        return circle_im

    if fill_mode == "vals":
        check.array_t(vals, shape=(locs.shape[0],))
        circle_im = np.zeros_like(im)
        for loc_i, (loc, val, keep) in enumerate(zip(locs, vals, keep_mask)):
            if keep:
                imops.set_with_mask_in_place(circle_im, brim, val, loc=loc, center=True)
        return circle_im


# In development
# -------------------------------------------------------------------------------------


def plot_sigproc_stats(run):
    # Hist quality, peaks per field, background, etc.
    # Maybe done as rows per field heatmap
    z = ZPlots()
    with z(_cols=4, f_plot_width=250, f_plot_height=280):
        fields_df = run.sigproc_v1.fields()
        z.hist(
            fields_df.quality,
            f_x_axis_label="quality",
            f_y_axis_label="n_frames",
            _bins=np.linspace(0, 400, 100),
        )
        by_quality = run.sigproc_v1.fields().sort_values(by="quality")

        def get(i):
            row = by_quality.iloc[i]
            im = run.sigproc_v1.raw_im(row.field_i, row.channel_i, row.cycle_i)
            return im, row

        best_im, best = get(-1)
        worst_im, worst = get(0)
        median_im, median = get(run.sigproc_v1.n_frames // 2)
        cspan = (0, np.percentile(median_im.flatten(), q=99, axis=0))
        z.im(
            best_im,
            _cspan=cspan,
            f_title=f"Best field={best.field_i} channel={best.channel_i} cycle={best.cycle_i}",
        )
        z.im(
            median_im,
            _cspan=cspan,
            f_title=f"Median field={median.field_i} channel={median.channel_i} cycle={median.cycle_i}",
        )
        z.im(
            worst_im,
            _cspan=cspan,
            f_title=f"Worst field={worst.field_i} channel={worst.channel_i} cycle={worst.cycle_i}",
        )


def text_sigproc_overview(run):
    # Number of fields, Number of peaks,

    ch_map = " ".join(
        [
            f"{in_ch}->{run.sigproc_v1.params.input_channel_to_output_channel(in_ch)}"
            for in_ch in range(run.sigproc_v1.params.n_input_channels)
        ]
    )

    print(
        f"Sigproc ran over:\n"
        f"  {run.sigproc_v1.n_fields} fields\n"
        f"  {run.sigproc_v1.params.n_input_channels} input channels\n"
        f"Sigproc wrote:\n"
        f"  {run.sigproc_v1.params.n_output_channels} output channels\n"
        f"  {len(run.sigproc_v1.peaks()):,} peaks were found\n"
        f"The channel mapping was:\n"
        f"  {ch_map}\n"
    )


def plot_channel_signal_histograms(
    run, limit_cycles=None, limit_field=None, div_noise=False, **kw
):
    # Used in sigproc_template
    _x_ticks = Munch(
        precision=0, use_scientific=True, power_limit_high=0, power_limit_low=0
    )
    if "_x_ticks" not in kw:
        kw["_x_ticks"] = _x_ticks

    limit_cycles = (
        range(run.sigproc_v1.n_cycles) if limit_cycles is None else limit_cycles
    )
    n_bins = 100

    field_label = "(all fields)" if limit_field is None else f"field {limit_field}"

    def get_signal():
        signal = (
            run.sigproc_v1.sig()
            if limit_field is None
            else run.sigproc_v1.signal_radmat_for_field(limit_field)
        )
        if div_noise:
            noise = (
                run.sigproc_v1.noi()
                if limit_field is None
                else run.sigproc_v1.noise_radmat_for_field(limit_field)
            )
            signal = utils.np_safe_divide(signal, noise, default=0)

        return signal

    def _hist_ch_cy(ch, cy):
        ch_signal = get_signal()[:, slice(ch, ch + 1, 1), slice(cy, cy + 1, 1)]
        non_nan = ch_signal[~np.isnan(ch_signal)]

        # If there is no signal then add a zero entry to satisfy logic below.
        if non_nan.shape[0] == 0:
            non_nan = np.array([[0]])

        p99 = np.percentile(non_nan, 99)
        _hist, _edges = np.histogram(non_nan, bins=n_bins)

        # consider returning _hist[1:].max() below because the 0-bin is often dominating
        # and the shape of the rest of the data is generally more interesting?
        return non_nan, p99, _hist.max()

    # Establish uniform axes by sampling everything in a first pass
    max_x = 0
    max_y = 0
    for cy in limit_cycles:
        for ch in range(run.sigproc_v1.n_channels):
            _, x, y = _hist_ch_cy(ch, cy)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if kw.get("_range_only", 0):
        return max_x, max_y

    z = kw.get("_zplots_context", ZPlots())
    with z(
        _cols=None if "_cols" in kw else run.sigproc_v1.n_channels,
        f_plot_width=250,
        f_plot_height=250,
        f_x_range=kw.get("f_x_range", (0, max_x)),
        f_y_range=kw.get("f_y_range", (0, max_y * 1.1)),
    ):
        for cy in limit_cycles:
            for ch in range(run.sigproc_v1.n_channels):
                data, _, _ = _hist_ch_cy(ch, cy)
                z.hist(
                    data,
                    _bins=n_bins,
                    f_title=f"CH{ch} cycle {cy} {field_label}",
                    **kw,
                )

    return max_x, max_y


def wizard_df_filter(run, limit_columns=None):
    """
    Wizard to explore sigprocv2 data as a a large table

    Audience:
        Trained technicians.

    Goal:
        Allow user to see:
            Any anomalies that are occuring as a result of the stage position
    """
    import qgrid  # Defer slow imports

    if limit_columns is not None:
        df = run.sigproc_v1.all_df()[limit_columns]
    else:
        df = run.sigproc_v1.fields__n_peaks__peaks__radmat()

    return qgrid.show_grid(df, show_toolbar=True, precision=2, grid_options=dict())


def wizard_xy_df(
    run,
    channel_i=None,
    result_block="sigproc_v2",
    ignore_fields=None,
    red_bottom=False,
    **kwargs,
):
    """
    Wizard to explore sigprocv2 data as a function of the stage

    Audience:
        Trained technicians.

    Goal:
        Allow user to see:
            Any anomalies that are occuring as a result of the stage position

    Options:
        ignore_fields allows you to remove certain fields from
        consideration because they mess up the auto-scale.
    """
    from ipywidgets import interact  # Defer slow imports
    from bokeh.models import ColorBar  # Defer slow imports
    from bokeh.plotting import figure, ColumnDataSource, show  # Defer slow imports
    from bokeh.transform import linear_cmap  # Defer slow imports
    from bokeh.palettes import Viridis256

    stage_df = (
        run.ims_import.metadata()
        .groupby("field_i")[["stage_x", "stage_y"]]
        .mean()
        .reset_index()
    )

    df = run[result_block].fields__n_peaks__peaks__radmat()
    if result_block == "sigproc_v1":
        df = df.drop(["stage_x", "stage_y"], axis=1)

    if ignore_fields is not None:
        stage_df = stage_df[~stage_df.field_i.isin(ignore_fields)]
        df = df[~df.field_i.isin(ignore_fields)]

    df = df.set_index("field_i").join(stage_df.set_index("field_i"))

    if "stage_x" not in df.columns:
        print("No stage information in sigprocv2 data.  Tif import without metadata?")
        return

    # If there are any fields that had no peaks found, many of the numeric fields that we
    # may want to plot will contain NaN and generate an exception below.  To allow the
    # viz to highlight the problem, fill these values (e.g. n_peaks, etc) with 0
    # If 0 turns out to be inappropriate for some columns, pass a df of values for cols.
    # See below where I have made color 0 bright red to highlight these.
    df = df.fillna(0)

    if channel_i is not None:
        df = df[df.channel_i == channel_i].reset_index()

    heat_name_wid = displays.dropdown(df, "", "n_peaks")

    def heat(heat_name):
        try:
            field_grp = df.groupby(["field_i"])
            means_df = field_grp[["stage_x", "stage_y", heat_name]].mean()
            min_ = means_df[heat_name].min()
            max_ = means_df[heat_name].max()
            if max_ == min_:
                # ZBS: I hit a problem where the eps caused an exception
                # Just chaning it to 1 for now
                max_ = min_ + 1  # np.finfo(float).eps
            val = (255 * (means_df[heat_name] - min_) / (max_ - min_)).astype(int)
            lut = np.array(Viridis256)
            colors = lut[val]

            source = ColumnDataSource(means_df)

            # TODO?: I wanted values set to 0 via df.fillna(0) above to be highlighted, and in general
            # you can't tell apart "the lowest" from other low values in these heatmaps.  So I made the first
            # color bright red - but that means that all items that fall into the lowest of the
            # 256 bins will get colored red.  Maybe this is not a good solution, but as I play
            # with the heatmap using this scheme, I find that it is drawing my attention to
            # some interesting things, so I'm leaving it for now.
            palette = list(Viridis256)
            if red_bottom:
                palette[0] = "#FF0000"

            mapper = linear_cmap(
                field_name=heat_name, palette=palette, low=min_, high=max_
            )
            f = figure(
                match_aspect=True,
                tooltips=displays.tooltips(df),
                plot_width=800,
                plot_height=800,
            )
            f.rect(
                # Note these x/y values are where the rect is centered - are the stage?
                x="stage_x",
                y="stage_y",
                width=90,
                height=90,
                source=source,
                color=mapper,
            )
            color_bar = ColorBar(
                color_mapper=mapper["transform"], width=8, location=(0, 0)
            )
            f.add_layout(color_bar, "right")
            show(f)
        except Exception as e:
            debug(e)

    interact(heat, heat_name=heat_name_wid)


def wizard_scat_df(
    run,
    default_x="field_i",
    default_y="signal",
    channel_i=None,
    result_block="sigproc_v2",
):
    """
    Wizard to explore sigprocv2 data on any pivot.

    Audience:
        Trained technicians.

    Goal:
        Allow user to see:
            Any pivot version of the sigprocv2 data
    """
    from ipywidgets import interact  # Defer slow imports
    from bokeh.plotting import figure, ColumnDataSource, show  # Defer slow imports

    df = run[result_block].fields__n_peaks__peaks__radmat()
    if channel_i is not None:
        df = df[df.channel_i == channel_i].reset_index()
    x_name_wid = displays.dropdown(df, "X:", default_x)
    y_name_wid = displays.dropdown(df, "Y:", default_y)

    def scat(x_name, y_name, x_noise):
        n_peaks = df.shape[0]
        n_samples = 3000
        mask = data.subsample(np.arange(n_peaks), n_samples)
        _df = df.loc[mask].copy()
        _df[x_name] = _df[x_name] + np.random.uniform(-x_noise, +x_noise, size=len(_df))
        source = ColumnDataSource(_df)
        f = figure(
            tooltips=displays.tooltips(df),
            x_axis_label=x_name,
            y_axis_label=y_name,
            plot_width=800,
            plot_height=800,
        )
        f.scatter(x=x_name, y=y_name, fill_alpha=0.5, line_color=None, source=source)
        show(f)

    interact(scat, x_name=x_name_wid, y_name=y_name_wid, x_noise=0.1)


def wizard_raw_images(
    run,
    max_bright=1_000,
    show_circles=True,
    peak_i_square=True,
    square_radius=4,
    cycle_stride=1,
    horizontal_layout=False,
    result_block="sigproc_v2",
):
    """
    Wizard to explore raw images

    Audience:
        Trained technicians.

    Goal:
        Allow user to see:
            Raw images
    """

    from ipywidgets import interact_manual  # Defer slow imports

    res = run[result_block]
    df = res.fields__n_peaks__peaks()

    def show_raw(peak_i, field_i, channel_i, cycle_i, max_bright, show_circles):
        field_i = int(field_i) if field_i != "" else None
        channel_i = int(channel_i)
        cycle_i = int(cycle_i)
        if field_i is None:
            peak_i = int(peak_i)
            peak_records = df[df.peak_i == peak_i]
            field_i = int(peak_records.iloc[0].field_i)
        else:
            peak_i = None

        all_sig = res.sig()

        # mask_rects_for_field = res.raw_mask_rects_df()[field_i]
        # Temporarily removed. This is going to involve some groupby Super-Pandas-Kungfu(tm)
        # Here is my too-tired start...
        """
        import pandas as pd
        df = pd.DataFrame([
            (0, 0, 0, 100, 110, 120, 130),
            (0, 0, 1, 101, 111, 121, 131),
            (0, 0, 2, 102, 112, 122, 132),
            (0, 1, 0, 200, 210, 220, 230),
            (0, 1, 1, 201, 211, 221, 231),
            (0, 1, 2, 202, 212, 222, 232),
            (1, 0, 0, 1100, 1110, 1120, 1130),
            (1, 0, 1, 1101, 1111, 1121, 1131),
            (1, 0, 2, 1102, 1112, 1122, 1132),
            (1, 1, 0, 1200, 1210, 1220, 1230),
            (1, 1, 1, 1201, 1211, 1221, 1231),
        ], columns=["frame_i", "ch_i", "cy_i", "x", "y", "w", "h"])

        def rec(row):
            return row[["x", "y"]]

        df.set_index("frame_i").groupby(["frame_i"]).apply(rec)
        """
        mask_rects_for_field = None

        cspan = (0, max_bright)
        circle = cspan[1] * imops.generate_donut_mask(4, 3)
        square = cspan[1] * imops.generate_square_mask(square_radius)

        z = ZPlots()
        sig_for_channel = all_sig[:, channel_i, :]
        sig_top = np.median(sig_for_channel) + np.percentile(sig_for_channel, 99.9)

        if peak_i is not None:
            rad = sig_for_channel[peak_i]
            rad = rad.reshape(1, rad.shape[0])
            print(
                "\n".join(
                    [
                        f"    cycle {cycle:2d}: {r:6.0f}"
                        for cycle, r in enumerate(rad[0])
                    ]
                )
            )
            z.scat(x=range(len(rad[0])), y=rad[0])
            z.im(rad, _cspan=(0, sig_top), f_plot_height=50, _notools=True)

            # This is inefficient because the function we will call
            # does the same image load, but I'd prefer to not repeat
            # the code here and want to be able to call this fn
            # from notebooks:
            _raw_peak_i_zoom(
                field_i,
                res,
                df,
                peak_i,
                channel_i,
                zoom=3.0,
                square_radius=square_radius,
                x_pad=1,
                cspan=cspan,
                separate=False,
                show_circles=show_circles,
            )

        if result_block == "sigproc_v1":
            im = res.raw_chcy_ims(field_i).copy()[channel_i, cycle_i]
        else:
            im = res.aln_ims[field_i, channel_i, cycle_i].copy()

        if peak_i is not None:
            cy_rec = peak_records[peak_records.cycle_i == cycle_i].iloc[0]
            im_marker = square if peak_i_square else circle
            imops.accum_inplace(
                im, im_marker, loc=XY(cy_rec.raw_x, cy_rec.raw_y), center=True,
            )

        elif show_circles:
            peak_records = df[(df.field_i == field_i) & (df.cycle_i == cycle_i)]

            # In the case of a field with no peaks, n_peaks may be NaN, so check that we have
            # some peaks before passing NaNs to imops.
            if peak_records.n_peaks.iloc[0] > 0:
                for i, peak in peak_records.iterrows():
                    imops.accum_inplace(
                        im, circle, loc=XY(peak.raw_x, peak.raw_y), center=True,
                    )

        z.im(
            im,
            f_title=f"ch_i={channel_i}  cy_i={cycle_i}  fl_i={field_i}",
            _full=True,
            _noaxes=True,
            _cspan=(0, float(max_bright)),
        )
        displays.fix_auto_scroll()

    interact_manual(
        show_raw,
        peak_i="1",
        field_i="",
        channel_i="0",
        cycle_i="0",
        max_bright=max_bright,
        show_circles=show_circles,
    )


def _raw_peak_i_zoom(
    field_i,
    res,
    df,
    peak_i,
    channel=0,
    zoom=3.0,
    square_radius=7,
    x_pad=0,
    cspan=(0, 5_000),
    separate=False,
    show_circles=True,
):
    peak_i = int(peak_i)
    peak_records = df[df.peak_i == peak_i]
    field_i = int(peak_records.iloc[0].field_i)

    im = res.raw_chcy_ims(field_i)
    all_sig = res.sig()

    square = cspan[1] * imops.generate_square_mask(square_radius)

    sig_for_channel = all_sig[peak_i, channel, :]
    sig_top = np.median(sig_for_channel) + np.percentile(sig_for_channel, 99.9)

    height = (square_radius + 1) * 2 + 1
    one_width = height + x_pad
    all_width = one_width * res.n_cycles - x_pad
    im_all_cycles = np.zeros((height, all_width))

    f_plot_height = height * zoom
    f_plot_width = all_width * zoom

    z = ZPlots()
    if show_circles:
        for cycle_i in range(res.n_cycles):
            im_with_marker = np.copy(im[channel, cycle_i])
            cy_rec = peak_records[peak_records.cycle_i == cycle_i].iloc[0]
            loc = XY(cy_rec.raw_x, cy_rec.raw_y)

            imops.accum_inplace(im_with_marker, square, loc=loc, center=True)
            im_with_marker = imops.extract_with_mask(
                im_with_marker,
                imops.generate_square_mask(square_radius + 1, True),
                loc=loc,
                center=True,
            )
            imops.accum_inplace(
                im_all_cycles, im_with_marker, loc=XY(cycle_i * one_width, 0)
            )

            if separate:
                z.im(
                    im_with_marker,
                    _noaxes=True,
                    f_plot_height=int(f_plot_height),
                    f_plot_width=int(f_plot_height),
                    _notools=True,
                    f_match_aspect=True,
                    _cspan=cspan,
                )

    if not separate:
        z.im(
            im_all_cycles,
            _noaxes=True,
            f_plot_height=int(f_plot_height),
            f_plot_width=int(f_plot_width),
            _notools=True,
            f_match_aspect=True,
            _cspan=cspan,
        )
