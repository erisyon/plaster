cimport c_common as c

cdef extern from "c_nn_v2_fast.h":
    ctypedef struct NNV2FastContext:
        c.Size n_neighbors
        c.Size n_cols

        c.Table test_unit_radmat
        c.Table train_dyetrack_weights
        c.Table train_dyemat
        c.Table train_dyepeps
        c.Table train_dye_i_to_dyepep_offset
        c.Table output_pred_pep_iz
        c.Table output_pred_dye_iz
        c.Table output_scores
        c.ProgressFn progress_fn

        c.Size n_threads
        c.Size n_rows
        c.Index next_row_i
        c.Size n_rows_per_block

    void context_start(NNV2FastContext *ctx)
    void context_free(NNV2FastContext *ctx)
