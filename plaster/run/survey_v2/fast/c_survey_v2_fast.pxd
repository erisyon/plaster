cdef extern from "c_survey_v2_fast.h":
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

    ctypedef struct Context:
        Table dyemat
        Table dyepeps
        Table pep_i_to_dyepep_row_i
        Table dyt_i_to_mlpep_i
        Index next_pep_i
        Size n_threads
        Size n_peps
        Size n_neighbors
        Size n_dyts
        Size n_dyt_cols
        Table output_pep_i_to_isolation_metric
        Float32 distance_to_assign_an_isolated_pep


    ctypedef void (*ProgressFn)(int complete, int total, int retry)
    void context_work_orders_start(Context *ctx)
