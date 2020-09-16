from munch import Munch
import numpy as np
import pandas as pd
from zest import zest
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.tools.log.log import debug
from plaster.tools.utils.utils import npf, np_array_same
from plaster.tools.schema import check


def zest_v2_all_df():
    props = Munch(
        radmat=npf(
            [
                [
                    [[3.0, 2.0, 1.0], [1.0, 0.0, 1.0]],
                    [[4.0, 2.0, 1.0], [1.0, 1.0, 0.0]],
                    [[5.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
                    [[6.0, 3.0, 0.9], [1.0, 0.0, 1.0]],
                ],
                [
                    [[0.3, 0.2, 0.1], [0.3, 0.2, 0.1]],
                    [[0.4, 0.2, 0.1], [0.3, 0.2, 0.1]],
                    [[0.5, 0.2, 0.1], [0.3, 0.2, 0.1]],
                    [[0.6, 0.3, 2.9], [0.3, 0.2, 0.1]],
                ],
            ],
        ),
        peak_df=pd.DataFrame(
            [
                (0, 0, 0, 1.0, 11.0),
                (0, 0, 1, 2.0, 12.0),
                (0, 1, 0, 3.0, 13.0),
                (0, 1, 1, 4.0, 14.0),
            ],
            columns=["peak_i", "field_i", "field_peak_i", "aln_x", "aln_y"],
        ),
        field_df=pd.DataFrame(
            [
                (0, 0, 0, 1, 2, 0.1, 100.0, 2, 10, 5, 10, 500, 20, 490),
                (0, 0, 1, 2, 3, 0.2, 110.0, 2, 10, 5, 10, 500, 20, 490),
                (0, 0, 2, 3, 4, 0.3, 120.0, 2, 10, 5, 10, 500, 20, 490),
                (0, 1, 0, 4, 5, 0.4, 130.0, 2, 10, 5, 10, 500, 20, 490),
                (0, 1, 1, 5, 6, 0.5, 140.0, 2, 10, 5, 10, 500, 20, 490),
                (0, 1, 2, 6, 7, 0.6, 150.0, 2, 10, 5, 10, 500, 20, 490),
                (1, 0, 0, 7, 8, 0.7, 160.0, 2, 10, 6, 20, 510, 30, 480),
                (1, 0, 1, 8, 9, 0.8, 170.0, 2, 10, 6, 20, 510, 30, 480),
                (1, 0, 2, 9, 0, 0.9, 180.0, 2, 10, 6, 20, 510, 30, 480),
                (1, 1, 0, 0, 1, 0.0, 190.0, 2, 10, 6, 20, 510, 30, 480),
                (1, 1, 1, 1, 2, 0.1, 200.0, 2, 10, 6, 20, 510, 30, 480),
                (1, 1, 2, 2, 3, 0.2, 210.0, 2, 10, 6, 20, 510, 30, 480),
            ],
            columns=[
                "field_i",
                "channel_i",
                "cycle_i",
                "shift_y",
                "shift_x",
                "aln_score",
                "bg_median",
                "n_mask_rects",
                "mask_area",
                "border_size",
                "aligned_roi_l",
                "aligned_roi_r",
                "aligned_roi_b",
                "aligned_roi_t",
            ],
        ),
        mask_rects_df=pd.DataFrame(
            [
                (0, 0, 0, 1, 2, 3, 4),
                (0, 0, 1, 2, 2, 3, 4),
                (0, 0, 2, 3, 2, 3, 4),
                (0, 1, 0, 4, 2, 3, 4),
                (0, 1, 1, 5, 2, 3, 4),
                (0, 1, 2, 6, 2, 3, 4),
                (1, 0, 0, 7, 2, 3, 4),
                (1, 0, 1, 8, 2, 3, 4),
                (1, 0, 2, 9, 2, 3, 4),
                (1, 1, 0, 0, 2, 3, 4),
                (1, 1, 1, 1, 2, 3, 4),
                (1, 1, 2, 2, 2, 3, 4),
            ],
            columns=["field_i", "channel_i", "cycle_i", "l", "r", "w", "h",],
        ),
    )

    # ramar got refactored and I just reorder it here
    props["radmat"] = np.moveaxis(props["radmat"], 0, 3)

    def _mock_load_field_prop(inst, field_i, prop):
        return props[prop]

    zest.stack_mock(
        SigprocV2Result._load_field_prop,
        substitute_fn=_mock_load_field_prop,
        reset_before_each=False,
    )

    res = SigprocV2Result(
        is_loaded_result=True, field_files=[""], n_peaks=4, n_channels=2, n_cycles=3
    )

    def it_np_signal_radmat():
        assert np_array_same(res.sig(), props.radmat[:, :, :, 0])

    def it_np_noise_radmat():
        assert np_array_same(res.noi(), props.radmat[:, :, :, 1])

    # All of the following are testing the DataFrames

    def it_fields():
        assert res.fields().equals(props.field_df)

    def it_radmats():
        rad_df = res.radmats()
        check.df_t(rad_df, SigprocV2Result.radmat_df_schema)
        assert len(rad_df) == 4 * 2 * 3

        # Sanity check a few
        assert (
            rad_df[
                (rad_df.peak_i == 1) & (rad_df.channel_i == 1) & (rad_df.cycle_i == 1)
            ].signal.values[0]
            == 1.0
        )
        assert (
            rad_df[
                (rad_df.peak_i == 2) & (rad_df.channel_i == 0) & (rad_df.cycle_i == 0)
            ].signal.values[0]
            == 5.0
        )

    # def it_mask_rects():
    #     rects_df = res.mask_rects()
    #     check.df_t(rects_df, SigprocV2Result.mask_rects_df_schema)
    #     assert len(rects_df) == 2 * 2 * 3

    # TASK: Lots of work left here

    # def it_peaks():
    #     assert res.peaks()[["field_i", "field_peak_i", "aln_x", "aln_y"]].equals(
    #         props.peak_df[["field_i", "field_peak_i", "aln_x", "aln_y"]]
    #     )
    #     assert np.all(res.peaks().peak_i.values == np.arange(4))

    # def it_radmats__peaks():
    #     df = res.radmats__peaks()
    #
    #     # Sanity check a few
    #     assert df[(df.peak_i == 1) & (df.channel_i == 0) & (df.cycle_i == 0)].signal.values[0] == 4.0
    #     assert df[(df.peak_i == 3) & (df.channel_i == 0) & (df.cycle_i == 2)].signal.values[0] == 0.9
    #     assert np.all(df[df.peak_i == 3].aln_x.values == 4.0)
    #     assert np.all(df[df.peak_i == 3].aln_y.values == 14.0)
    #
    # def it_n_peaks():
    #     df = res.n_peaks()
    #     assert np.all(df.n_peaks.values == 2)
    #
    # def it_field__n_peaks__peaks():
    #     df = res.field__n_peaks__peaks()
    #     raise NotImplementedError
    #
    # def it_fields__n_peaks__radmat__peaks():
    #     df = res.fields__n_peaks__radmat__peaks()
    #     debug(df)
    #     raise NotImplementedError

    zest()
