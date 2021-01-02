"""
Functions related to the analysis of classified sigproc calls.
I can't say that this code feels well developed yet.
Likely it will evolve.
"""
import numpy as np
import pandas as pd
from plaster.run.sigproc_v2.sigproc_v2_result import df_to_radmat
from plaster.tools.utils import utils


def join_sigproc_classify(sigproc_v2_result, nn_v2_result):
    """
    Make a "sigcalls" DF which is a union of sigproc_v2_result and nn_v2_result
    """
    dyemat = nn_v2_result._dyemat
    n_dyts, n_cycles = dyemat.shape
    dyt_df = pd.DataFrame(
        dict(
            dyt_c=dyemat.flatten(),
            dyt_i=np.repeat(np.arange(n_dyts), n_cycles),
            cycle_i=np.tile(np.arange(n_cycles), (n_dyts,)),
        )
    )

    call_df = nn_v2_result.calls("sigproc", include_nul_calls=True)

    sig_df = sigproc_v2_result.fields__n_peaks__peaks__radmat()
    df = sig_df.set_index("peak_i", drop=False).join(
        call_df.set_index("peak_i"), how="left"
    )
    df.dyt_i = np.nan_to_num(df.dyt_i).astype(int)
    df = (
        df.set_index(["dyt_i", "cycle_i"])
        .join(dyt_df.set_index(["dyt_i", "cycle_i"]))
        .reset_index()
    )
    df = df.sort_values(["peak_i", "cycle_i"]).reset_index(drop=True)
    return df


def cycle_balance(sig, dyemat):
    """
    sig and dyemat are the same length.
    Typically sig will be row balanced already
    Returns:
        A scalar for each cycle that when multiplied by the signal brings the signal into balance
    """
    assert sig.shape[0] == dyemat.shape[0]
    one_mask = dyemat == 1
    cy_bal = np.nanmean(np.where(one_mask, sig, np.nan), axis=0)
    return np.mean(cy_bal) / cy_bal


def sigcalls_to_sigmats(
    sigcalls_df,
    filt_mask,
    dyemat,
    ch_i=0,
    cy_bal=None,
    sort_by_dyt=False,
    max_aspect_ratio=1.5,
    row_bal=True,
):
    n_peaks = sigcalls_df.peak_i.max() + 1
    n_cycles = dyemat.shape[1]

    sig = df_to_radmat(sigcalls_df, channel_i=ch_i)[:, :]
    filt_mask = filt_mask.reshape(n_peaks, n_cycles)
    unsorted_filt_rows = np.all(filt_mask, axis=1)
    unsorted_df = (
        sigcalls_df[["peak_i", "dyt_i", "k", "logp_dyt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    unsorted_asr = df_to_radmat(
        sigcalls_df, radmat_field="aspect_ratio", channel_i=ch_i
    )[:, :]
    unsorted_k = df_to_radmat(sigcalls_df, radmat_field="k", channel_i=ch_i)[:, :]
    unsorted_max_asr = np.max(
        np.where(dyemat[unsorted_df.dyt_i] > 0, unsorted_asr, 0.0), axis=1
    )
    unsorted_filt_rows = unsorted_filt_rows & (unsorted_max_asr < max_aspect_ratio)
    unsorted_dyt = dyemat[unsorted_df.dyt_i][unsorted_filt_rows]

    if sort_by_dyt:
        sorted_df = unsorted_df.sort_values("dyt_i").reset_index(drop=True)
        sorted_filt_rows = unsorted_filt_rows[sorted_df.peak_i]
        sorted_sig_raw = sig[sorted_df.peak_i][sorted_filt_rows]
        dyt = dyemat[sorted_df.dyt_i][sorted_filt_rows]
        if row_bal:
            with utils.np_no_warn():
                sig_rw_bal = sorted_sig_raw / sorted_df.k[sorted_filt_rows, None]
        else:
            sig_rw_bal = sorted_sig_raw
        k = unsorted_k[sorted_df.peak_i][sorted_filt_rows]
        sig_raw = sorted_sig_raw
        peak_iz = sorted_df[sorted_filt_rows].peak_i.values
    else:
        unsorted_sig_raw = sig[unsorted_df.peak_i][unsorted_filt_rows]
        dyt = unsorted_dyt
        if row_bal:
            with utils.np_no_warn():
                sig_rw_bal = unsorted_sig_raw / unsorted_df.k[unsorted_filt_rows, None]
        else:
            sig_rw_bal = unsorted_sig_raw
        k = unsorted_k[unsorted_filt_rows]
        sig_raw = unsorted_sig_raw
        peak_iz = unsorted_df[unsorted_filt_rows].peak_i.values

    if cy_bal is None:
        cy_bal = np.ones((n_cycles))

    return sig_rw_bal * cy_bal, dyt, k, sig_raw, peak_iz


def n_dyts_by_count(cnt, dyemat):
    """
    Used to "triangle" dyemats, return the dyt_i for the last of a given count.
    """
    return np.argmin(np.all(dyemat <= cnt, axis=1)) - 1
