cdef extern from "c_nn_v2_fast.h":
    ctypedef unsigned long Size
    ctypedef unsigned long Index
    ctypedef unsigned char DyeType
    ctypedef float RadType
    ctypedef float Score

    ctypedef struct Context:
        Size n_cols
        Size radmat_n_rows
        RadType *radmat

        Size train_dyetracks_n_rows
        DyeType *train_dyetracks
        Size train_dyepeps_n_rows
        DyePepRec *train_dyepeps

        CallRec *output_callrecs

    void context_start(Context *ctx)
    void context_free(Context *ctx)
