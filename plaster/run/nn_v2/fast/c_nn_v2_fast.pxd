cdef extern from "c_nn_v2_fast.h":
    ctypedef unsigned char Uint8
    ctypedef unsigned long Uint64
    ctypedef unsigned long Size
    ctypedef unsigned long Index
    ctypedef unsigned long HashKey
    ctypedef unsigned char DyeType
    ctypedef unsigned char CycleKindType
    ctypedef unsigned long PIType
    ctypedef double RecallType
    ctypedef unsigned int Size32
    ctypedef unsigned int Index32
    ctypedef float RadType
    ctypedef float Score
    ctypedef float WeightType

    ctypedef struct Table:
        Uint8 *rows
        Uint64 n_bytes_per_row
        Uint64 n_max_rows
        Uint64 n_rows

    Table table_init(Uint8 *base, Size n_bytes, Size n_bytes_per_row)

    ctypedef struct DyePepRec:
        Size count
        Index dtr_i
        Index pep_i

    ctypedef struct Context:
        Size n_neighbors
        Size n_cols

        Table test_unit_radmat
        Table train_dyetrack_weights
        Table train_dyemat
        Table output_pred_iz
        Table output_scores

    void context_start(Context *ctx)
    void context_free(Context *ctx)
