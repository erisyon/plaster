cimport c_common as c

cdef extern from "c_nn_v2_fast.h":
    ctypedef struct NNV2FastContext:
        c.Size n_neighbors
        c.Size n_cols
        c.Uint64 run_against_all_dyetracks

        c.Tab test_radmat
        c.Tab train_dyetrack_weights
        c.Tab train_dyemat
        c.Tab train_dyepeps
        c.Tab train_dye_i_to_dyepep_offset
        c.Tab output_pred_pep_iz
        c.Tab output_pred_dye_iz
        c.Tab output_scores
        c.ProgressFn progress_fn
        c.CheckKeyboardInterruptFn check_keyboard_interrupt_fn

        c.Size n_threads
        c.Size n_rows
        c.Index next_row_i
        c.Size n_rows_per_block

    int context_start(NNV2FastContext *ctx)
    void context_free(NNV2FastContext *ctx)
