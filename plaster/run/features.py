import numpy as np
import pandas as pd
from plaster.tools.log.log import debug


def features(run, ch_i=0):
    """
    Extract standard features for every peak
    """
    df = run.sigproc_v2.peaks()

    # Merge in stage metadata
    stage_df = (
        run.ims_import.metadata()[["field_i", "stage_x", "stage_y"]]
        .groupby("field_i")
        .mean()
    )
    df = pd.merge(left=df, right=stage_df, left_on="field_i", right_on="field_i")
    df["flowcell_x"] = df.stage_x + df.aln_x
    df["flowcell_y"] = df.stage_y + df.aln_y

    sig = run.sigproc_v2.sig()[:, ch_i, :]
    assert sig.shape[0] == len(df)

    asr = run.sigproc_v2.aspect_ratio()[:, ch_i, :]
    assert asr.shape[0] == len(df)

    has_neighbor_stats = run.sigproc_v2.has_neighbor_stats()
    if has_neighbor_stats:
        nei = run.sigproc_v2.neighborhood_stats()
        assert nei.shape[0] == len(df)

    # Convenience aliases
    n_cycles = run.sigproc_v2.n_cycles
    n_peaks = df.peak_i.max() + 1
    im_mea = run.ims_import.dim

    # Contants needed for feature extraction
    dark = run.sigproc_v2.dark_estimate(ch_i, fields=None, n_sigmas=4.0)

    run_len = n_cycles - np.argmax(sig[:, ::-1] > dark, axis=1)
    row_iz, col_iz = np.indices(sig.shape)
    sig_run = np.where(col_iz < run_len[:, None], sig, np.nan)
    asr_run = np.where(col_iz < run_len[:, None], asr, np.nan)

    if has_neighbor_stats:
        nei_mean_run = np.where(col_iz < run_len[:, None], nei[:, :, 0], np.nan)
        nei_std_run = np.where(col_iz < run_len[:, None], nei[:, :, 1], np.nan)
        nei_median_run = np.where(col_iz < run_len[:, None], nei[:, :, 2], np.nan)
        nei_iqr_run = np.where(col_iz < run_len[:, None], nei[:, :, 3], np.nan)
    else:
        nei_mean_run = np.zeros((n_peaks, n_cycles))
        nei_std_run = np.zeros((n_peaks, n_cycles))
        nei_median_run = np.zeros((n_peaks, n_cycles))
        nei_iqr_run = np.zeros((n_peaks, n_cycles))

    df["radius"] = np.sqrt(
        (df.aln_x - im_mea // 2) ** 2 + (df.aln_y - im_mea // 2) ** 2
    )

    df["run_len"] = run_len
    df["run_med"] = np.nanmedian(sig_run, axis=1)
    df["run_men"] = np.nanmean(sig_run, axis=1)
    df["run_std"] = np.nanstd(sig_run, axis=1)
    df["run_iqr"] = np.subtract(*np.nanpercentile(sig_run, [75, 25], axis=1))
    df["run_max"] = np.nanmax(sig_run, axis=1)
    df["run_min"] = np.nanmin(sig_run, axis=1)
    df["run_rng"] = df.run_max - df.run_min

    df["nei_men"] = np.nanmean(nei_mean_run, axis=1)
    df["nei_std"] = np.nanmean(nei_std_run, axis=1)
    df["nei_med"] = np.nanmean(nei_median_run, axis=1)
    df["nei_iqr"] = np.nanmean(nei_iqr_run, axis=1)

    df["sig_med"] = np.median(sig, axis=1)
    df["sig_men"] = np.mean(sig, axis=1)
    df["sig_std"] = np.std(sig, axis=1)
    df["sig_iqr"] = np.subtract(*np.nanpercentile(sig, [75, 25], axis=1))
    df["sig_max"] = np.max(sig, axis=1)
    df["sig_min"] = np.min(sig, axis=1)
    df["sig_rng"] = df.sig_max - df.sig_min

    df["asr_med"] = np.nanmedian(asr_run, axis=1)
    df["asr_std"] = np.nanstd(asr_run, axis=1)
    df["asr_max"] = np.nanmax(asr_run, axis=1)
    df["asr_cy0"] = asr[:, 0]

    return df
