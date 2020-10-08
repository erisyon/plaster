import numba
import numpy as np


@numba.jit(nopython=True)
def from_true_pred(true, pred, true_dim, pred_dim):
    assert true.ndim == 1 and pred.ndim == 1
    # This assert can take upwards of 50ms, and given that this fn is called as an inner loop, we can save some time if we omit it
    # assert np.all((0 <= true) & (true < true_dim) & (0 <= pred) & (pred < pred_dim))
    index = (pred * true_dim + true).astype(np.int64)
    return np.reshape(
        np.bincount(index, minlength=true_dim * pred_dim), (pred_dim, true_dim)
    )


def conf_mat(df, prep_result_n_peps, mask=None):
    """
    Build a confusion matrix from the call bag.

    If the set_size parameters are not given it
    will determine those sizes by asking the prep_result.
    """
    true = df["true_pep_iz"].values
    pred = df["pred_pep_iz"].values

    # Compute true_set_size and pred_set_size if they are not specified
    true_set_size = prep_result_n_peps
    pred_set_size = prep_result_n_peps

    n_rows = len(df)
    if mask is not None:
        pred = np.copy(pred)
        pred[~mask] = 0

    return from_true_pred(true, pred, true_set_size, pred_set_size)


def conf_mat_at_score_threshold(df, score_thresh, scores, prep_result_n_peps):
    return conf_mat(
        df, prep_result_n_peps, mask=scores >= score_thresh.astype(scores.dtype)
    )


def scale_by_abundance(arr, abundance):
    """
        DHW 9/28/2020 - I profiled the check.array_t and the assert and in practice the impact appears minimal (<1ms in my test case)
        """
    assert np.all((abundance >= 1.0) | (abundance == 0.0))
    return arr * abundance.astype(int)


def _auc(x, y):
    """A simple rectangular (Euler) integrator. Simpler and easier than sklearn metrics"""
    zero_padded_dx = np.concatenate(([0], x))
    return (np.diff(zero_padded_dx) * y).sum()


def precision(arr):
    diag = np.diag(arr)
    sum_rows = np.sum(arr, axis=1)
    return np.divide(diag, sum_rows, out=np.zeros(arr.shape[0]), where=sum_rows != 0)


def recall(arr):
    diag = np.diag(arr)
    sum_cols = np.sum(arr, axis=0)
    return np.divide(diag, sum_cols, out=np.zeros(arr.shape[0]), where=sum_cols != 0)


def pr_curve_by_pep_with_abundance_inner_loop(
    df,
    step_size,
    n_steps,
    scores,
    pep_abundance,
    prsa,
    prep_result_n_peps,
    pep_iz,
    n_peps,
):
    precision_column = 0
    recall_column = 1

    for prsa_i, score_thresh in enumerate(np.linspace(1 - step_size, 0, n_steps)):
        # TODO: could opimize this by subselecting pep_iz for conf_mat if we're not
        # doing all peps - creates smaller confusion matrix.
        conf_mat = conf_mat_at_score_threshold(
            df, score_thresh, scores, prep_result_n_peps
        )
        assert pep_abundance is not None

        conf_mat = scale_by_abundance(conf_mat, pep_abundance)
        p = precision(conf_mat)[pep_iz]
        r = recall(conf_mat)[pep_iz]
        auc = np.array(
            [
                _auc(
                    prsa[0 : prsa_i + 1, p_i, recall_column],
                    prsa[0 : prsa_i + 1, p_i, precision_column],
                )
                for p_i in range(n_peps)
            ]
        )
        prsa[prsa_i] = np.transpose([p, r, [score_thresh] * n_peps, auc])


def pr_curve_by_pep_with_abundance_v2(
    df,
    step_size,
    n_steps,
    scores,
    pep_abundance,
    prsa,
    prep_result_n_peps,
    pep_iz,
    n_peps,
):
    pass


@numba.jit(nopython=True)
def pr_curve_by_pep_with_abundance_v2_inner_loop():
    """
    Used functions supported by numba:
    enumerate
    np.linspace (only the 3-argument form)
    """
    for prsa_i, score_thresh in enumerate(np.linspace(1 - step_size, 0, n_steps)):
        precision = np.zeros(prep_result_n_peps)
        recall = np.zeros(prep_result_n_peps)
        for i in range(prep_result_n_peps):
            # Build precision and recall directly rather than building the conf mat first
            pass
