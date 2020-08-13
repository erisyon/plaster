cdef extern from "c_survey_v2_fast.h":
    ctypedef struct Context:
        Table dyemat
        Table dyepeps

    ctypedef struct Table:
        Uint8 *rows
        Uint64 n_bytes_per_row
        Uint64 n_max_rows
        Uint64 n_rows

    Table table_init(Uint8 *base, Size n_bytes, Size n_bytes_per_row)
    Table table_init_readonly(Uint8 *base, Size n_bytes, Size n_bytes_per_row)

    void context_work_orders_start(Context *ctx)
