# TODO: DRY this with similar in other fast
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
    Table table_init_readonly(Uint8 *base, Size n_bytes, Size n_bytes_per_row)
    void table_set_row(Table *table, Index row_i, void *src)

    ctypedef struct DyePepRec:
        Index dtr_i
        Index pep_i
        Size count

    ctypedef void (*ProgressFn)(int complete, int total, int retry)

    ctypedef struct Context:
        Size n_neighbors
        Size n_cols

        Table test_unit_radmat
        Table train_dyetrack_weights
        Table train_dyemat
        Table train_dyepeps
        Table train_dye_i_to_dyepep_offset
        Table output_pred_pep_iz
        Table output_pred_dye_iz
        Table output_scores
        ProgressFn progress_fn

        Size n_threads
        Size n_rows
        Index next_row_i
        Size n_rows_per_block


    void context_start(Context *ctx)
    void context_free(Context *ctx)
