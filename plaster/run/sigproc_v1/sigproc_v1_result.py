"""
Sigproc results are generated in parallel by field.

At save time, the radmats for all fields is composited into one big radmat.
"""
import pandas as pd
import itertools
from plumbum import local
from plaster.tools.schema import check
from munch import Munch
import numpy as np
from plaster.tools.utils import utils
from plaster.tools.image.coord import Rect
from plaster.run.base_result import BaseResult
from plaster.run.sigproc_v1.sigproc_v1_params import SigprocV1Params
from plaster.tools.log.log import debug


class SigprocV1Result(BaseResult):
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

    name = "sigproc_v1"
    filename = "sigproc_v1.pkl"

    # fmt: off
    required_props = dict(
        params=SigprocV1Params,
        n_input_channels=int,
        n_channels=int,
        n_cycles=int,
    )

    peak_df_schema = dict(
        peak_i=int,
        field_i=int,
        field_peak_i=int,
        aln_x=int,
        aln_y=int,
    )

    field_df_schema = dict(
        field_i=int,
        channel_i=int,
        cycle_i=int,
        shift_y=int,
        shift_x=int,
        aln_score=float,
        bg_median=float,
        n_mask_rects=int,
        mask_area=int,
        border_size=int,
        quality=float,
        aligned_roi_rect_l=int,
        aligned_roi_rect_r=int,
        aligned_roi_rect_b=int,
        aligned_roi_rect_t=int,
    )

    radmat_df_schema = dict(
        peak_i=int,
        channel_i=int,
        cycle_i=int,
        signal=float,
        noise=float,
        localbg=float,
    )

    mask_rects_df_schema = dict(
        field_i=int,
        channel_i=int,
        cycle_i=int,
        l=int,
        r=int,
        w=int,
        h=int,
    )
    # fmt: on

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
        rows = [
            (field_i, ch, cy, rect[0], rect[1], rect[2], rect[3])
            for ch, cy_rects in enumerate(kwargs.pop("raw_mask_rects"))
            for cy, rects in enumerate(cy_rects)
            for i, rect in enumerate(rects)
        ]
        kwargs["mask_rects_df"] = pd.DataFrame(
            rows, columns=["field_i", "channel_i", "cycle_i", "l", "r", "w", "h"]
        )

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
            radmat = self.signal_radmat()
            np.save(
                str(self._folder / "full_signal_radmat.npy"), radmat, allow_pickle=False
            )

        super().save()

    def __repr__(self):
        try:
            return f"SigprocResult with files in {self._folder} {self.n_fields}"
        except Exception as e:
            return "SigprocResult"

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
    def n_frames(self):
        return (
            self.n_fields
            * self.params.n_output_channels
            * np.max(self.fields().cycle_i)
            + 1
        )

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

    def _load_ndarray_prop_from_all_fields(self, prop):
        """
        Stack the ndarray that is in prop along all fields
        """
        val = self._cache(prop)
        if val is None:
            val = np.vstack(
                [
                    self._load_field_prop(field_i, prop)
                    for field_i in range(self.n_fields)
                ]
            )
            self._cache(prop, val)
        return val

    # ndarray returns
    # ----------------------------------------------------------------

    def signal_radmat_for_field(self, field_i):
        return self._load_field_prop(field_i, "signal_radmat")

    def signal_radmat(self):
        """
        Returns ndarray(n_peaks, n_channels, n_cycles)
        """
        return self._load_ndarray_prop_from_all_fields("signal_radmat")

    def flat_radmat(self):
        """
        Returns ndarray(n_peaks, n_channels * n_cycles)
        """
        arr = self._load_ndarray_prop_from_all_fields("signal_radmat")
        assert arr.ndim == 3
        return arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))

    def noise_radmat_for_field(self, field_i):
        return self._load_field_prop(field_i, "noise_radmat")

    def noise_radmat(self):
        """
        Returns ndarray(n_peaks, n_channels, n_cycles)
        """
        return self._load_ndarray_prop_from_all_fields("noise_radmat")

    def localbg_radmat_for_field(self, field_i):
        return self._load_field_prop(field_i, "localbg_radmat")

    def localbg_radmat(self):
        """
        Returns ndarray(n_peaks, n_channels, n_cycles)
        """
        return self._load_ndarray_prop_from_all_fields("localbg_radmat")

    def raw_chcy_ims(self, field_i):
        filename = self._field_filename(field_i, is_debug=True)
        return utils.indexed_pickler_load(filename, prop_list="_raw_chcy_ims")

    def aligned_composite_bg_removed_im(self, field_i):
        filename = self._field_filename(field_i, is_debug=True)
        return utils.indexed_pickler_load(
            filename, prop_list="_aligned_composite_bg_removed_im"
        )

    def raw_im(self, field_i, channel_i, cycle_i):
        return self.raw_chcy_ims(int(field_i))[int(channel_i), int(cycle_i)]

    # DataFrame returns
    # ----------------------------------------------------------------

    def fields(self):
        df = self._load_df_prop_from_all_fields("field_df")
        check.df_t(df, self.field_df_schema, allow_extra_columns=True)
        return df

    def peaks(self):
        df = self._load_df_prop_from_all_fields("peak_df")
        check.df_t(df, self.peak_df_schema)

        # The peaks have a local frame_peak_i but they
        # don't have their pan-field peak_i set yet.
        df = df.reset_index(drop=True)
        df.peak_i = df.index

        return df

    def radmats(self):
        """
        Unwind a radmat into a giant dataframe with peak, channel, cycle
        """
        sigs = self.signal_radmat()
        nois = self.noise_radmat()
        lcbg = self.localbg_radmat()

        signal = sigs.reshape((sigs.shape[0] * sigs.shape[1] * sigs.shape[2]))
        noise = nois.reshape((nois.shape[0] * nois.shape[1] * nois.shape[2]))
        localbg = lcbg.reshape((lcbg.shape[0] * lcbg.shape[1] * lcbg.shape[2]))

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
                localbg=localbg,
            )
        )

    def mask_rects(self):
        df = self._load_df_prop_from_all_fields("mask_rects_df")
        check.df_t(df, self.mask_rects_df_schema)
        return df

    def radmats__peaks(self):
        return (
            self.radmats()
            .set_index("peak_i")
            .join(self.peaks().set_index("peak_i"))
            .reset_index()
        )

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

    def fields__n_peaks__peaks(self):
        """
        Add a "raw_x" "raw_y" position for each peak. This is the
        coordinate of the peak relative to the original raw image
        so that circles can be used to
        """
        fc_index = ["field_i"]

        df = (
            self.fields()
            .set_index("field_i")
            .join(self.n_peaks().set_index("field_i"), how="outer")
            .reset_index()
        )

        df = (
            df.set_index(fc_index)
            .join(self.peaks().set_index(fc_index), how="outer")
            .reset_index()
        )

        df["raw_x"] = df.aln_x + (df.shift_x - df.border_size)
        df["raw_y"] = df.aln_y + (df.shift_y - df.border_size)

        return df

    def fields__n_peaks__peaks__radmat(self):
        """
        Build a giant joined dataframe useful for debugging.
        The masked_rects are excluded from this as they clutter it up.
        """
        pcc_index = ["peak_i", "channel_i", "cycle_i"]

        df = (
            self.fields__n_peaks__peaks()
            .set_index(pcc_index)
            .join(
                self.radmats__peaks().set_index(pcc_index)[
                    ["signal", "noise", "localbg"]
                ]
            )
            .reset_index()
        )

        return df
