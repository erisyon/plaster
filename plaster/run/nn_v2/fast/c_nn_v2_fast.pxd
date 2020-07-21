cdef extern from "c_nn_v2_fast.h":
    ctypedef unsigned long Uint64
    ctypedef unsigned long Size
    ctypedef unsigned int Size32
    ctypedef unsigned long Index
    ctypedef unsigned int Index32
    ctypedef unsigned char DyeType
    ctypedef float RadType
    ctypedef float Score
    ctypedef float WeightType

    ctypedef struct DyePepRec:
        Size count
        Index dtr_i
        Index pep_i

    ctypedef struct Context:
        Size n_neighbors
        Size n_cols

        Size test_unit_radmat_n_rows
        RadType *test_unit_radmat

        Size train_dyemat_n_rows
        RadType *train_dyemat
        WeightType *train_dyetrack_weights

        Index32 *output_pred_iz
        Score *output_scores

    void context_start(Context *ctx)
    void context_free(Context *ctx)
