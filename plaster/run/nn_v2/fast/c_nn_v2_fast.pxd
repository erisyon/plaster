cdef extern from "c_nn_v2_fast.h":
    ctypedef unsigned long Size
    ctypedef unsigned int Size32
    ctypedef unsigned long Index
    ctypedef unsigned int Index32
    ctypedef unsigned char DyeType
    ctypedef float RadType
    ctypedef float Score

    ctypedef struct DyePepRec:
        Size dtr_i
        Size pep_i
        Size count

    ctypedef struct Context:
        Size n_neighbors
        Size n_cols
        Size radmat_n_rows
        RadType *radmat

        Size train_dyetracks_n_rows
        DyeType *train_dyetracks
        Size train_dyepeps_n_rows
        DyePepRec *train_dyepeps

        Index32 *output_pred_iz
        Score *output_scores

    void context_start(Context *ctx)
    void context_free(Context *ctx)
