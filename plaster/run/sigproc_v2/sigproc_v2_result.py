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
from plaster.run.calib.calib import Calib
from plaster.tools.utils import data
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
        calib=Calib,
        focus_per_field_per_channel=(type(None), list),
    )

    peak_df_schema = OrderedDict(
        peak_i=int,
        field_i=int,
        field_peak_i=int,
        aln_y=float,
        aln_x=float,
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
        aln_y=float,
        aln_x=float,
        focus_adjustment=float,
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
        self._cache_ims = {}

    def __repr__(self):
        try:
            return f"SigprocV2Result with files in {self._folder} with {self.n_fields} fields"
        except Exception as e:
            return "SigprocV2Result"

    def limit(self, field_i_start=0, n_field_files=1):
        self.field_files = self.field_files[
            field_i_start : field_i_start + n_field_files
        ]

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
        # TODO! Do not use the field list, rather save the n_fields
        # as a prop because if I re-run )(esp. during debugging)
        # and change the n_fields_limit then it still picks up the
        # old number of fields!
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

    def _load_df_prop_from_fields(self, prop, field_iz=None):
        """
        Stack the DF that is in prop along all fields
        """
        if field_iz is None:
            field_iz = tuple(range(self.n_fields))

        cache_key = f"{prop}_field_iz_{field_iz}"

        val = self._cache(cache_key)
        if val is None:
            dfs = [self._load_field_prop(field_i, prop) for field_i in field_iz]

            # If you concat an empty df with others, it will wreak havoc
            # on your column dtypes (e.g. int64->float64)
            non_empty_dfs = [df for df in dfs if len(df) > 0]

            val = pd.concat(non_empty_dfs, sort=False)
            self._cache(cache_key, val)
        return val

    def _fields_to_field_iz(self, fields):
        """
        fields is a union type:
            None: All fields
            int: a specific field
            slice: a slice of fields
            tuple: a list of specific fields
        """
        if fields is None:
            return tuple(list(range(self.n_fields)))
        elif isinstance(fields, int):
            return tuple([fields])
        elif isinstance(fields, slice):
            start = fields.start or 0
            stop = fields.stop or self.n_fields
            return tuple(list(range(start, stop, fields.step)))
        elif isinstance(fields, (tuple, list)):
            return tuple(fields)

        raise TypeError(
            f"fields of unknown type in _load_ndarray_prop_from_fields. {type(fields)}"
        )

    def _load_list_prop_from_fields(self, fields, prop):
        return [
            self._load_field_prop(field_i, prop)
            for field_i in self._fields_to_field_iz(fields)
        ]

    def _load_ndarray_prop_from_fields(self, fields, prop, vstack=True):
        """
        Stack the ndarray that is in prop along all fields
        """
        list_ = self._load_list_prop_from_fields(fields, prop)
        if vstack:
            val = np.vstack(list_)
        else:
            val = np.stack(list_)

        return val

    # list returns
    # ----------------------------------------------------------------

    def cy_locs_per_field(self, fields=None):
        """
        Return per-cycle peak locations in list form.

        Note that this function does not attempt to reconcile these lists
        and they are only used to try to track if there are a lot
        of anomalous stray peaks coming and going.

        Example:

        [  # Field 0
            [  # Cycle 0
                [101, 231],  # Found peak 0
                [210, 123],  # Found peak 1
            ],
            [  # Cycle 1
                [101, 231],  # Found peak 0
                [213, 123],  # Found peak 1
            ],
        ],
        [  # Field 1
            [  # Cycle 0
                [101, 231],  # Found peak 0
                [210, 123],  # Found peak 1
            ],
            [  # Cycle 1
                [101, 231],  # Found peak 0
                [213, 123],  # Found peak 1
            ],
        ],
        """
        return self._load_list_prop_from_fields(fields, "cy_locs")

    # ndarray returns
    # ----------------------------------------------------------------

    def locs(self, fields=None):
        """Return peak locations in array form"""
        df = self.peaks()
        field_iz = self._fields_to_field_iz(fields)
        df = df[df.field_i.isin(field_iz)]
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
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "radmat")[:, :, :, 2],
                **kwargs,
            )
        )

    def aspect_ratio(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "radmat")[:, :, :, 3],
                **kwargs,
            )
        )

    def _aln_chcy_ims(self, field_i, key):
        cache_key = (field_i, key)
        if cache_key not in self._cache_ims:
            filename = self._field_filename(field_i, is_debug=True)
            self._cache_ims[cache_key] = utils.indexed_pickler_load(
                filename, prop_list=key
            )
        return self._cache_ims[cache_key]

    def aln_filt_chcy_ims(self, field_i):
        return self._aln_chcy_ims(field_i, "_aln_filt_chcy_ims")

    def aln_unfilt_chcy_ims(self, field_i):
        return self._aln_chcy_ims(field_i, "_aln_unfilt_chcy_ims")

    def raw_chcy_ims(self, field_i):
        # Only for compatibility with wizard_raw_images
        # this can be deprecated once sigproc_v1 is deprecated
        return self.aln_filt_chcy_ims(field_i)

    @property
    def aln_ims(self):
        return FancyIndexer(
            (self.n_fields, self.n_channels, self.n_cycles),
            lookup_fn=lambda fl, ch, cy: self.aln_filt_chcy_ims(fl)[ch, cy],
        )

    @property
    def aln_unfilt_ims(self):
        return FancyIndexer(
            (self.n_fields, self.n_channels, self.n_cycles),
            lookup_fn=lambda fl, ch, cy: self.aln_unfilt_chcy_ims(fl)[ch, cy],
        )

    def fitmat(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "fitmat"), **kwargs,
            )
        )

    def neighborhood_stats(self, fields=None, **kwargs):
        return np.nan_to_num(
            self.flat_if_requested(
                self._load_ndarray_prop_from_fields(fields, "neighborhood_stats"),
                **kwargs,
            )
        )

    def has_neighbor_stats(self):
        nei_stat = self._load_field_prop(0, "neighborhood_stats")
        return nei_stat is not None

    # DataFrame returns
    # ----------------------------------------------------------------

    @disk_memoize()
    def fields(self, fields=None):
        df = self._load_df_prop_from_fields(
            "field_df", field_iz=self._fields_to_field_iz(fields)
        )
        check.df_t(df, self.field_df_schema, allow_extra_columns=True)
        return df

    @disk_memoize()
    def peaks(self, fields=None, n_peaks_subsample=None):
        df = self._load_df_prop_from_fields(
            "peak_df", field_iz=self._fields_to_field_iz(fields)
        )
        check.df_t(df, self.peak_df_schema)

        if self._has_prop("peak_fit_df"):
            fit_df = self._load_df_prop_from_fields(
                "peak_fit_df", field_iz=self._fields_to_field_iz(fields)
            )
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
        df = self._load_df_prop_from_fields("peak_fit_df")
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
        df = self._load_df_prop_from_fields("mask_rects_df")
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
    def fields__n_peaks__peaks(self, fields=None, n_peaks_subsample=None):
        """
        Add a "raw_x" "raw_y" position for each peak. This is the
        coordinate of the peak relative to the original raw image
        so that circles can be used to
        """
        df = (
            self.fields(fields=fields)
            .set_index("field_i")
            .join(self.n_peaks().set_index("field_i"), how="left")
            # This was previously using how="outer". I change it
            # to left when I added the field_iz argument and I'm not
            # sure this isn't going to break something
            .rename(columns=dict(aln_y="field_aln_y", aln_x="field_aln_x"))
            .reset_index()
        )

        pdf = self.peaks(n_peaks_subsample=n_peaks_subsample)

        df = df.set_index("field_i").join(pdf.set_index("field_i")).reset_index()
        # This join was previously inverted (pdf join fields)
        # but I reversed it when I added the field_iz argument

        df["raw_x"] = df.aln_x - (df.field_aln_x)
        df["raw_y"] = df.aln_y - (df.field_aln_y)

        return df

    @disk_memoize()
    def fields__n_peaks__peaks__radmat(self, fields=None, n_peaks_subsample=None):
        """
        Build a giant joined dataframe useful for debugging.
        The masked_rects are excluded from this as they clutter it up.
        field_iz if specified must be a tuple (for memoize)
        """
        pcc_index = ["peak_i", "channel_i", "cycle_i"]

        df = (
            self.fields__n_peaks__peaks(
                fields=fields, n_peaks_subsample=n_peaks_subsample
            )
            .set_index(pcc_index)
            .join(
                self.radmats__peaks().set_index(pcc_index)[
                    ["signal", "noise", "snr", "aspect_ratio"]
                ],
            )
            .reset_index()
        )

        return df

    def has_new_locs(self):
        return (
            self._has_prop("cy_locs")
            and self.cy_locs_per_field(fields=0)[0] is not None
        )

    @disk_memoize()
    def new_locs(self, fields=None):
        """
        Returns a dataframe of peak locations that we found in that
        cycle that did not reconcile with cycle 0 locs.

        DataFrame(field_i, cycle_i, aln_y, aln_x)
        """
        from scipy.spatial.distance import cdist

        new_locs_df = pd.DataFrame()
        for fl_i in self._fields_to_field_iz(fields):
            # Index [0] in following because the fields always returns a list even if the index is scalar
            cy_locs = self.cy_locs_per_field(fields=fl_i)[0]
            if cy_locs is not None:

                cy0_locs = self.locs(fields=fl_i)

                for cy_i, cy_locs in enumerate(cy_locs):
                    if cy_i == 0:
                        continue

                    # Find closest old loc (axis=1) to each new loc (axis=0)
                    dists = cdist(cy_locs, cy0_locs, "euclidean")
                    closest_i = np.argmin(dists, axis=1)
                    closest_d = dists[np.arange(cy_locs.shape[0]), closest_i]

                    # Any new loc (axis=0) that is > 1.5 pixels from an old is a new loc
                    new_locs_mask = closest_d > 1.5
                    new_cy_locs = cy_locs[new_locs_mask]
                    n_new_cy_locs = len(new_cy_locs)

                    cy_df = pd.DataFrame(
                        dict(
                            field_i=[fl_i] * n_new_cy_locs,
                            cycle_i=[cy_i] * n_new_cy_locs,
                            aln_y=new_cy_locs[:, 0],
                            aln_x=new_cy_locs[:, 1],
                        )
                    )
                    new_locs_df = new_locs_df.append(cy_df)

        return new_locs_df

    def dark_estimate(self, ch_i, fields=None, n_sigmas=4.0):
        """
        Use last cycle of (least likely to be polluted with signal)
        and estimate the width of the darkness distribution
        by using a one-sided std.
        """
        sig_last_cy = self.sig(fields=fields)[:, ch_i, -1]
        zero_sigma = data.one_sided_nanstd(sig_last_cy)
        dark = n_sigmas * zero_sigma
        return dark

    def asr_threshold(self):
        # This is a guess based on various simulations
        # and I'm making this a function in case at some
        # point I choose to convert this is a distribution-based
        # technique.
        return 1.3


# The following operate on dataframes returned by fields__n_peaks__peaks__radmat


def mean_non_dark_asr(df, dark, ch_i):
    """
    Return the mean of non-dark aspect ratios per row
    """
    assert dark is not None
    rad_pt = pd.pivot_table(
        df, values="signal", index=["peak_i"], columns=["channel_i", "cycle_i"]
    )
    ch_rad_pt = rad_pt.loc[:, ch_i]
    asr_pt = pd.pivot_table(
        df, values="aspect_ratio", index=["peak_i"], columns=["channel_i", "cycle_i"]
    )
    ch_asr_pt = asr_pt.loc[:, ch_i]
    non_dark_asr = np.where(ch_rad_pt >= dark, ch_asr_pt, np.nan)
    with utils.np_no_warn():
        return np.nanmean(non_dark_asr, axis=1)


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
            non_dark_asr = np.where(ch_rad_pt >= dark, ch_asr_pt, np.nan)
            with utils.np_no_warn():
                mean_asr_per_row = np.nanmean(non_dark_asr, axis=1)

            if max_aspect_ratio is not None:
                keep_peaks_mask &= mean_asr_per_row <= max_aspect_ratio

            if min_aspect_ratio is not None:
                keep_peaks_mask &= mean_asr_per_row >= min_aspect_ratio

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
        _df = keep_df.join(df.set_index("peak_i", drop=False))

    return _df


def df_to_radmat(
    df,
    radmat_field="signal",
    channel_i=None,
    n_cycles=None,
    nan_to_zero=True,
    nan_missing_rows=False,
    n_peaks=None,
):
    """
    Convert the dataframe filtered by df_filter into a radmat

    channel_i:
        If non-None pluck out that channel
    n_cycles:
        If non None specifies the cycles otherwise uses df.cycle_i.max() + 1
    nan_to_zero:
        Convert nans to zeros
    nan_missing_rows:
        adding nan rows where the peak_i is missing in the df.
        If used, n_peaks must be non-None
    n_peaks:
        Used with nan_missing_rows
    """
    if n_cycles is None:
        n_cycles = df.cycle_i.max() + 1

    if channel_i is None:
        n_channels = df.channel_i.max() + 1
    else:
        n_channels = 1

    if nan_missing_rows:
        new_index = pd.MultiIndex.from_arrays(
            [
                np.tile(np.arange(0, n_peaks), n_cycles),
                np.repeat(np.arange(0, n_cycles), n_peaks),
            ],
            names=("peak_i", "cycle_i"),
        )
        df = (
            df.set_index(["peak_i", "cycle_i"])
            .reindex(new_index)
            .reset_index()
            .sort_values(["peak_i", "cycle_i"])
            .reset_index(drop=True)
        )

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
