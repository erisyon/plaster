import numpy as np
from plaster.tools.c_common import c_common
from plumbum import local, FG
import ctypes as c


class NNV2Context(c.Structure):
    _fixup_fields = [
        # Input Tables
        ("train_dyemat", "Tab"),
        ("train_dyepeps", "Tab"),
        ("test_radmat", "Tab"),
        # Parameters
        ("beta", "Float64"),
        ("sigma", "Float64"),
        ("zero_mu", "Float64"),
        ("zero_sigma", "Float64"),
        ("row_k_std", "Float64"),
        # Options
        ("n_neighbors", "Size"),
        ("run_row_k_fit", "Bool"),
        ("run_against_all_dyetracks", "Bool"),
        # Derived properties
        ("n_rows", "Size"),
        ("n_cols", "Size"),
        ("train_dyetrack_weights", "Tab"),
        ("train_dyt_i_to_dyepep_offset", "Tab"),
        ("progress_fn", "ProgressFn"),
        # Outputs
        ("output_pred_pep_iz", "Tab"),
        ("output_pred_dyt_iz", "Tab"),
        ("output_scores", "Tab"),
        ("output_dists", "Tab"),
        # Internal fields
        ("_work_order_lock", "pthread_mutex_t"),
        ("_flann_index_lock", "pthread_mutex_t"),
        ("_pyfunction_lock", "pthread_mutex_t"),
    ]


_lib = None

recompile = True


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    with local.cwd("/erisyon/plaster/plaster/run/nn_v2/c"):
        if recompile:
            with local.env(DST_FOLDER="/erisyon/plaster/plaster/run/nn_v2/c"):
                local["./build.sh"] & FG
        lib = c.CDLL("./_nn_v2.so")

    lib.context_init.argtypes = [
        c.POINTER(NNV2Context),  # NNV2FastContext *ctx
    ]

    lib.classify_radrows.argtypes = [
        c_common.typedef_to_ctype("Index"),  # Index radrow_start_i,
        c_common.typedef_to_ctype("Size"),  # Size n_radrows,
        c.POINTER(NNV2Context),  # NNV2FastContext *ctx
    ]

    _lib = lib
    return lib


def context_create(
    train_dyemat,
    train_dyepeps,
    test_radmat,
    beta,
    sigma,
    zero_beta,
    zero_sigma,
    row_k_std,
    n_neighbors=8,
    run_row_k_fit=False,
    run_against_all_dyetracks=False,
):
    return NNV2Context(
        # train_dyemat=Tab(train_dyemat),
        # train_dyepeps=Tab(train_dyepeps),
        # test_radmat=Tab(test_radmat),
        beta=beta,
        sigma=sigma,
        zero_beta=zero_beta,
        zero_sigma=zero_sigma,
        row_k_std=row_k_std,
        n_neighbors=n_neighbors,
        run_row_k_fit=run_row_k_fit,
        run_against_all_dyetracks=run_against_all_dyetracks,
        n_cols=train_dyemat.shape[1],
        # train_dyt_weights=Tab(?),
        # train_dyt_i_to_dyepep_offset=Tab(?),
        # progress_fn=progress_fn,
        # output=Tab(?),
    )


def classify_radrows(radrow_start_i, n_radrows, nn_v2_context):
    lib = load_lib()
    lib.classify_radrows()
