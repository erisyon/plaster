cimport c_common as c

cdef extern from "c_survey_v2_fast.h":
    ctypedef struct SurveyV2FastContext:
        c.Tab dyemat
        c.Tab dyepeps
        c.Tab pep_i_to_dyepep_row_i
        c.Tab dyt_i_to_mlpep_i
        c.Tab output_pep_i_to_isolation_metric
        c.Index next_pep_i
        c.Size n_threads
        c.Size n_peps
        c.Size n_neighbors
        c.Size n_dyts
        c.Size n_dyt_cols
        c.Float32 distance_to_assign_an_isolated_pep
        c.ProgressFn progress_fn

    void context_start(SurveyV2FastContext *ctx)
