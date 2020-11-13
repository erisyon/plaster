"""
Sigproc results are generated in parallel by field.

At save time, the radmats for all fields is composited into one big radmat.
"""
import pandas as pd
import itertools
import warnings
from collections import OrderedDict
from plumbum import local
from plaster.tools.schema import check
from plaster.tools.image.coord import ROI, YX, HW
import numpy as np
from plaster.tools.utils import utils
from plaster.tools.utils.fancy_indexer import FancyIndexer
from plaster.run.base_result import BaseResult, disk_memoize
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.calibration.calibration import Calibration
from plaster.tools.log.log import debug, prof


class SigprocV2Result(BaseResult):
    """
    Understanding alignment coordinates

    Each field has n_channels and n_cycles
    The channels are all aligned already (stage doesn't move between channels)
    But the stage does move between cycles and therefore an alignment is needed.
    The stack of cycle images are aligned in coordinates relative to the 0th cycles.
    The fields are stacked into a composite image large enough to hold the worst-case shift.
    Each field in the field_df has a shift_x, shift_y.
    The maximum absolute value of all of those shifts is called the border.
    The border is the amount added around all edges to accomdate all images.
    """

    name = "sigproc_v2"
    filename = "sigproc_v2.pkl"

    # fmt: off
    required_props = OrderedDict(
        # Note that these do not include props in the save_field
        params=SigprocV2Params,
        n_channels=(type(None), int),
        n_cycles=(type(None), int),
        calib=Calibration,
        focus_per_field_per_channel=(type(None), list),
    )

    peak_df_schema = OrderedDict(
        peak_i=int,
        field_i=int,
        field_peak_i=int,
        aln_y=int,
        aln_x=int,
    )

    peak_fit_df_schema = OrderedDict(
        peak_i=int,
        field_i=int,
        field_peak_i=int,
        amp=float,
        std_x=float,
        std_y=float,
        pos_x=float,
        pos_y=float,
        rho=float,
        const=float,
        mea=float,
    )

    field_df_schema = OrderedDict(
        field_i=int,
        channel_i=int,
        cycle_i=int,
        aln_y=int,
        aln_x=int,
        aln_score=float,
        # n_mask_rects=int,
        # mask_area=int,
        # quality=float,
        # aligned_roi_rect_l=int,
        # aligned_roi_rect_r=int,
        # aligned_roi_rect_b=int,
        # aligned_roi_rect_t=int,
    )

    radmat_df_schema = OrderedDict(
        peak_i=int,
        # field_i=int,
        # field_peak_i=int,
        channel_i=int,
        cycle_i=int,
        signal=float,
        noise=float,
        snr=float,
        aspect_ratio=float,
    )

    # mask_rects_df_schema = dict(
    #     field_i=int,
    #     channel_i=int,
    #     cycle_i=int,
    #     l=int,
    #     r=int,
    #     w=int,
    #     h=int,
    # )
    # fmt: on

    def __hash__(self):
        return hash(id(self))

    def _field_filename(self, field_i, is_debug):
        return self._folder / f"{'_debug_' if is_debug else ''}field_{field_i:03d}.ipkl"

    def save_field(self, field_i, **kwargs):
        """
        When using parallel field maps we can not save into the result
        because that will not be serialized back to the main thread.
        Rather, use temporary files and gather at save()

        Note that there is no guarantee of the order these are created.
        """

        # CONVERT raw_mask_rects to a DataFrame
        # rows = [
        #     (field_i, ch, cy, rect[0], rect[1], rect[2], rect[3])
        #     for ch, cy_rects in enumerate(kwargs.pop("raw_mask_rects"))
        #     for cy, rects in enumerate(cy_rects)
        #     for i, rect in enumerate(rects)
        # ]
        # kwargs["mask_rects_df"] = pd.DataFrame(
        #     rows, columns=["field_i", "channel_i", "cycle_i", "l", "r", "w", "h"]
        # )

        non_debug_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        utils.indexed_pickler_dump(
            non_debug_kwargs, self._field_filename(field_i, is_debug=False)
        )

        debug_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        utils.indexed_pickler_dump(
            debug_kwargs, self._field_filename(field_i, is_debug=True)
        )

    def save(self, save_full_signal_radmat_npy=False):
        """
        Extract the radmat from the fields and stack them in one giant mat
        """
        self.field_files = [i.name for i in sorted(self._folder // "field*.ipkl")]
        self.debug_field_files = [
            i.name for i in sorted(self._folder // "_debug_field*.ipkl")
        ]

        if save_full_signal_radmat_npy:
            radmat = self.sig()
            np.save(
                str(self._folder / "full_signal_radmat.npy"), radmat, allow_pickle=False
            )

        super().save()

    def __init__(self, folder=None, is_loaded_result=False, **kwargs):
        super().__init__(folder, is_loaded_result=is_loaded_result, **kwargs)
        self._cache_aln_chcy_ims = {}

    def __repr__(self):
        try:
            return f"SigprocV2Result with files in {self._folder} {self.n_fields}"
        except Exception as e:
            return "SigprocV2Result"

    def _cache(self, prop, val=None):
        # TASK: This might be better done with a yielding context
        cache_key = f"_load_prop_cache_{prop}"
        if val is not None:
            self[cache_key] = val
            return val
        cached = self.get(cache_key)
        if cached is not None:
            return cached
        return None

    @property
    def n_fields(self):
        return utils.safe_len(self.field_files)

    @property
    def n_cols(self):
        return self.n_cycles * self.n_channels

    @property
    def n_frames(self):
        return (
            self.n_fields
            * self.params.n_output_channels
            * np.max(self.fields().cycle_i)
            + 1
        )

    def fl_ch_cy_iter(self):
        return itertools.product(
            range(self.n_fields), range(self.n_channels), range(self.n_cycles)
        )

    def _has_prop(self, prop):
        # Assume field 0 is representative of all fields
        field_i = 0
        name = local.path(self.field_files[field_i]).name
        props = utils.indexed_pickler_load(
            self._folder / name, prop_list=[prop], skip_missing_props=True
        )
        return prop in props.keys()

    def _load_field_prop(self, field_i, prop):
        """Mockpoint"""
        name = local.path(self.field_files[field_i]).name
        return utils.indexed_pickler_load(self._folder / name, prop_list=prop)

    def _load_df_prop_from_all_fields(self, prop):
        """
        Stack the DF that is in prop along all fields
        """
        val = self._cache(prop)
        if val is None:
            dfs = [
                self._load_field_prop(field_i, prop) for field_i in range(self.n_fields)
            ]

            # If you concat an empty df with others, it will wreak havoc
            # on your column dtypes (e.g. int64->float64)
            non_empty_dfs = [df for df in dfs if len(df) > 0]

            val = pd.concat(non_empty_dfs, sort=False)
            self._cache(prop, val)
        return val

    def _fields_to_start_stop(self, fields):
        if fields is None:
            start = 0
            stop = self.n_fields
        elif isinstance(fields, slice):
            start = fields.start or 0
            stop = fields.stop or self.n_fields
            assert fields.step in (None, 1)
        elif isinstance(fields, int):
            start = fields
            stop = fields + 1
        else:
            raise TypeError(
                f"fields of unknown type in _load_ndarray_prop_from_fields. {type(fields)}"
            )
        return start, stop

    def _load_ndarray_prop_from_fields(self, fields, prop, vstack=True):
        """
        Stack the ndarray that is in prop along all fields
        """

        field_start, field_stop = self._fields_to_start_stop(fields)

        list_ = [
            self._load_field_prop(field_i, prop)
            for field_i in range(field_start, field_stop)
        ]

        if vstack:
            val = np.vstack(list_)
        else:
            val = np.stack(list_)

        return val

    # ndarray returns
    # ----------------------------------------------------------------

    def locs(self, fields=None):
        """Return peak locations in array form"""
        df = self.peaks()
        field_start, field_stop = self._fields_to_start_stop(fields)
        field_stop -= 1  # Because the following is inclusive
        df = df[df.field_i.between(field_start, field_stop, inclusive=True)]
        return df[["aln_y", "aln_x"]].values

    def flat_if_requested(self, mat, flat_chcy=False):
        if flat_chcy:
            return utils.mat_flatter(mat)
        return mat

    def sig(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "radmat")[:, :, :, 0],
                **kwargs,
            )
        )

    def noi(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "radmat")[:, :, :, 1],
                **kwargs,
            )
        )

    def snr(self, fields=None, **kwargs):
        return np.nan_to_num(
            utils.np_safe_divide(
                self.sig(fields=fields, **kwargs), self.noi(fields=fields, **kwargs)
            )
        )

    def aspect_ratio(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "radmat")[:, :, :, 2],
                **kwargs,
            )
        )

    def aln_chcy_ims(self, field_i):
        if field_i not in self._cache_aln_chcy_ims:
            filename = self._field_filename(field_i, is_debug=True)
            self._cache_aln_chcy_ims[field_i] = utils.indexed_pickler_load(
                filename, prop_list="_aln_chcy_ims"
            )
        return self._cache_aln_chcy_ims[field_i]

    def raw_chcy_ims(self, field_i):
        # Only for compatibility with wizard_raw_images
        # this can be deprecated once sigproc_v2 is deprecated
        return self.aln_chcy_ims(field_i)

    @property
    def aln_ims(self):
        return FancyIndexer(
            (self.n_fields, self.n_channels, self.n_cycles),
            lookup_fn=lambda fl, ch, cy: self.aln_chcy_ims(fl)[ch, cy],
        )

    def fitmat(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "fitmat"), **kwargs,
            )
        )

    def difmat(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "difmat"), **kwargs,
            )
        )

    def picmat(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "picmat"), **kwargs,
            )
        )

    def sftmat(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "sftmat"), **kwargs,
            )
        )

    # DataFrame returns
    # ----------------------------------------------------------------

    @disk_memoize()
    def fields(self):
        df = self._load_df_prop_from_all_fields("field_df")
        check.df_t(df, self.field_df_schema, allow_extra_columns=True)
        return df

    @disk_memoize()
    def peaks(self, n_peaks_subsample=None):
        df = self._load_df_prop_from_all_fields("peak_df")
        check.df_t(df, self.peak_df_schema)

        if self._has_prop("peak_fit_df"):
            fit_df = self._load_df_prop_from_all_fields("peak_fit_df")
            check.df_t(df, self.peak_fit_df_schema)
            df = df.set_index(["field_i", "field_peak_i"]).join(
                fit_df.set_index(["field_i", "field_peak_i"])
            )

        # The peaks have a local frame_peak_i but they
        # don't have their pan-field peak_i set yet.
        df = df.reset_index(drop=True)
        df.peak_i = df.index

        if n_peaks_subsample is not None:
            df = df.sample(n_peaks_subsample, replace=True)

        return df

    @disk_memoize()
    def peak_fits(self):
        df = self._load_df_prop_from_all_fields("peak_fit_df")
        check.df_t(df, self.peak_fit_df_schema)

        # The peaks have a local frame_peak_i but they
        # don't have their pan-field peak_i set yet.
        df = df.reset_index(drop=True)
        df.peak_i = df.index

        return df

    @disk_memoize()
    def radmats(self):
        """
        Unwind a radmat into a giant dataframe with peak, channel, cycle
        """
        sigs = self.sig()
        nois = self.noi()
        snr = self.snr()
        aspect_ratios = self.aspect_ratio()

        signal = sigs.reshape((sigs.shape[0] * sigs.shape[1] * sigs.shape[2]))
        noise = nois.reshape((nois.shape[0] * nois.shape[1] * nois.shape[2]))
        snr = snr.reshape((snr.shape[0] * snr.shape[1] * snr.shape[2]))
        aspect_ratio = aspect_ratios.reshape(
            (aspect_ratios.shape[0] * aspect_ratios.shape[1] * aspect_ratios.shape[2])
        )

        peaks = list(range(sigs.shape[0]))
        channels = list(range(self.n_channels))
        cycles = list(range(self.n_cycles))
        peak_cycle_channel_product = list(itertools.product(peaks, channels, cycles))
        peaks, channels, cycles = list(zip(*peak_cycle_channel_product))

        return pd.DataFrame(
            dict(
                peak_i=peaks,
                channel_i=channels,
                cycle_i=cycles,
                signal=signal,
                noise=noise,
                snr=snr,
                aspect_ratio=aspect_ratio,
            )
        )

    @disk_memoize()
    def mask_rects(self):
        df = self._load_df_prop_from_all_fields("mask_rects_df")
        check.df_t(df, self.mask_rects_df_schema)
        return df

    @disk_memoize()
    def radmats__peaks(self):
        return (
            self.radmats()
            .set_index("peak_i")
            .join(self.peaks().set_index("peak_i"))
            .reset_index()
        )

    @disk_memoize()
    def n_peaks(self):
        df = (
            self.peaks()
            .groupby("field_i")[["field_peak_i"]]
            .max()
            .rename(columns=dict(field_peak_i="n_peaks"))
            .reset_index()
        )
        df.n_peaks += 1
        return df

    @disk_memoize()
    def fields__n_peaks__peaks(self, n_peaks_subsample=None):
        """
        Add a "raw_x" "raw_y" position for each peak. This is the
        coordinate of the peak relative to the original raw image
        so that circles can be used to
        """
        df = (
            self.fields()
            .set_index("field_i")
            .join(self.n_peaks().set_index("field_i"), how="outer")
            .rename(columns=dict(aln_y="field_aln_y", aln_x="field_aln_x"))
            .reset_index()
        )

        pdf = self.peaks(n_peaks_subsample=n_peaks_subsample)

        df = pdf.set_index("field_i").join(df.set_index("field_i")).reset_index()

        df["raw_x"] = df.aln_x - (df.field_aln_x)
        df["raw_y"] = df.aln_y - (df.field_aln_y)

        return df

    @disk_memoize()
    def fields__n_peaks__peaks__radmat(self, n_peaks_subsample=None):
        """
        Build a giant joined dataframe useful for debugging.
        The masked_rects are excluded from this as they clutter it up.
        """
        pcc_index = ["peak_i", "channel_i", "cycle_i"]

        df = (
            self.fields__n_peaks__peaks(n_peaks_subsample=n_peaks_subsample)
            .set_index(pcc_index)
            .join(
                self.radmats__peaks().set_index(pcc_index)[
                    ["signal", "noise", "snr", "aspect_ratio"]
                ]
            )
            .reset_index()
        )

        return df


# The following operate on dataframes returned by fields__n_peaks__peaks__radmat


def df_filter(
    df,
    fields=None,
    reject_fields=None,
    roi=None,
    channel_i=0,
    dark=None,
    on_through_cy_i=None,
    off_at_cy_i=None,
    monotonic=None,
    min_intensity_cy_0=None,
    max_intensity_cy_0=None,
    max_intensity_any_cycle=None,
    min_intensity_per_cycle=None,
    max_intensity_per_cycle=None,
    max_aspect_ratio=None,
    min_aspect_ratio=None,
    radmat_field="signal",
    max_k=None,
    min_score=None,
):
    """
    A general filtering tool that operates on the dataframe returned by
    sigproc_v2.fields__n_peaks__peaks__radmat()
    """
    n_channels = df.channel_i.max() + 1

    # REMOVE unwanted fields
    if fields is None:
        fields = list(range(df.field_i.max() + 1))
    if reject_fields is not None:
        fields = list(filter(lambda x: x not in reject_fields, fields))
    _df = df[df.field_i.isin(fields)].reset_index(drop=True)

    # REMOVE unwanted peaks by ROI
    if roi is None:
        roi = ROI(YX(0, 0), HW(df.raw_y.max(), df.raw_x.max()))
    _df = _df[
        (roi[0].start <= _df.raw_y)
        & (_df.raw_y < roi[0].stop)
        & (roi[1].start <= _df.raw_x)
        & (_df.raw_x < roi[1].stop)
    ].reset_index(drop=True)

    if max_k is not None:
        _df = _df[_df.k <= max_k]

    if min_score is not None:
        _df = _df[_df.score >= min_score]

    # OPERATE on radmat if needed
    fields_that_operate_on_radmat = [
        dark,
        on_through_cy_i,
        off_at_cy_i,
        monotonic,
        min_intensity_cy_0,
        max_intensity_cy_0,
        max_intensity_any_cycle,
        min_intensity_per_cycle,
        max_intensity_per_cycle,
    ]

    if any([field is not None for field in fields_that_operate_on_radmat]):
        assert 0 <= channel_i < n_channels

        rad_pt = pd.pivot_table(
            _df, values=radmat_field, index=["peak_i"], columns=["channel_i", "cycle_i"]
        )
        ch_rad_pt = rad_pt.loc[:, channel_i]
        keep_peaks_mask = np.ones((ch_rad_pt.shape[0],), dtype=bool)

        if max_aspect_ratio is not None or min_aspect_ratio is not None:
            assert dark is not None
            asr_pt = pd.pivot_table(
                _df,
                values="aspect_ratio",
                index=["peak_i"],
                columns=["channel_i", "cycle_i"],
            )
            ch_asr_pt = asr_pt.loc[:, channel_i]
            if max_aspect_ratio is not None:
                keep_peaks_mask &= np.all(
                    (ch_asr_pt <= max_aspect_ratio) | (ch_rad_pt < dark), axis=1
                )
            if min_aspect_ratio is not None:
                keep_peaks_mask &= np.all(
                    (ch_asr_pt >= min_aspect_ratio) | (ch_rad_pt < dark), axis=1
                )

        if on_through_cy_i is not None:
            assert dark is not None
            keep_peaks_mask &= np.all(
                ch_rad_pt.loc[:, 0 : on_through_cy_i + 1] > dark, axis=1
            )

        if off_at_cy_i is not None:
            assert dark is not None
            keep_peaks_mask &= np.all(ch_rad_pt.loc[:, off_at_cy_i:] < dark, axis=1)

        if monotonic is not None:
            d = np.diff(ch_rad_pt.values, axis=1)
            keep_peaks_mask &= np.all(d < monotonic, axis=1)

        if min_intensity_cy_0 is not None:
            keep_peaks_mask &= ch_rad_pt.loc[:, 0] >= min_intensity_cy_0

        if max_intensity_cy_0 is not None:
            keep_peaks_mask &= ch_rad_pt.loc[:, 0] <= max_intensity_cy_0

        if max_intensity_any_cycle is not None:
            keep_peaks_mask &= np.all(
                ch_rad_pt.loc[:, :] <= max_intensity_any_cycle, axis=1
            )

        if min_intensity_per_cycle is not None:
            for cy_i, inten in enumerate(min_intensity_per_cycle):
                if inten is not None:
                    keep_peaks_mask &= ch_rad_pt.loc[:, cy_i] >= inten

        if max_intensity_per_cycle is not None:
            for cy_i, inten in enumerate(max_intensity_per_cycle):
                if inten is not None:
                    keep_peaks_mask &= ch_rad_pt.loc[:, cy_i] <= inten

        keep_peak_i = ch_rad_pt[keep_peaks_mask].index.values
        keep_df = pd.DataFrame(dict(keep_peak_i=keep_peak_i)).set_index("keep_peak_i")
        _df = (
            keep_df.join(df.set_index("peak_i"))
            .reset_index()
            .rename(columns=dict(index="peak_i"))
        )

    return _df


def df_to_radmat(
    df, radmat_field="signal", channel_i=None, n_cycles=None, nan_to_zero=True
):
    """
    Convert the dataframe filtered by df_filter into a radmat
    """
    if n_cycles is None:
        n_cycles = df.cycle_i.max() + 1

    if channel_i is None:
        n_channels = df.channel_i.max() + 1
    else:
        n_channels = 1

    radmat = pd.pivot_table(
        df,
        values=radmat_field,
        index=["field_i", "peak_i"],
        columns=["channel_i", "cycle_i"],
    )
    radmat = radmat.values
    n_rows = radmat.shape[0]
    radmat = radmat.reshape((n_rows, n_channels, n_cycles))

    if channel_i is not None:
        radmat = radmat[:, channel_i, :]

    if nan_to_zero:
        return np.nan_to_num(radmat)

    return radmat


def radmat_from_df_filter(
    df, radmat_field="signal", channel_i=None, return_df=False, **kwargs
):
    """
    Apply the filter args from df_filter and return a radmat
    """
    _df = df_filter(df, channel_i=channel_i, radmat_field=radmat_field, **kwargs)
    radmat = df_to_radmat(_df, channel_i=channel_i, radmat_field=radmat_field)

    if return_df:
        return radmat, _df
    else:
        return radmat
