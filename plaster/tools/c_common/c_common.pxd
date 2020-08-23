cdef extern from "c_common.h":
    ctypedef unsigned char Uint8
    ctypedef unsigned short Uint16
    ctypedef unsigned int Uint32
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
    ctypedef float Float32
    ctypedef double Float64
    ctypedef Float32 IsolationType

    Uint64 UINT64_MAX
    Uint64 N_MAX_CHANNELS
    Uint64 N_MAX_CYCLES
    Uint64 N_MAX_NEIGHBORS
    Uint8 NO_LABEL
    Uint8 CYCLE_TYPE_PRE
    Uint8 CYCLE_TYPE_MOCK
    Uint8 CYCLE_TYPE_EDMAN

    enum: TAB_NOT_GROWABLE
    enum: TAB_GROWABLE

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
        Size n_reads

    ctypedef void (*ProgressFn)(int complete, int total, int retry)

    ctypedef struct Tab:
        void *base
        Uint64 n_bytes_per_row
        Uint64 n_max_rows
        Uint64 n_rows
        int b_growable

    int sanity_check()

    void tab_tests()
    void tab_dump(Tab *tab, char *msg)
    Tab tab_subset(Tab *src, Index row_i, Size n_rows)
    Tab tab_by_n_rows(void *base, Size n_rows, Size n_bytes_per_row, int b_growable)
    Tab tab_by_size(void *base, Size n_bytes, Size n_bytes_per_row, int b_growable)
