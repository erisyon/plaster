cdef extern from "c_sim_v2_fast.h":
    ctypedef unsigned char Uint8
    ctypedef unsigned long Uint64
    ctypedef double Float64
    ctypedef unsigned long Size
    ctypedef unsigned long Index
    ctypedef unsigned long HashKey
    ctypedef unsigned char DyeType
    ctypedef unsigned char CycleKindType
    ctypedef unsigned long PIType
    ctypedef double RecallType

    Uint64 UINT64_MAX
    Uint64 N_MAX_CHANNELS
    Uint64 N_MAX_CYCLES
    Uint8 NO_LABEL
    Uint8 CYCLE_TYPE_PRE
    Uint8 CYCLE_TYPE_MOCK
    Uint8 CYCLE_TYPE_EDMAN

    ctypedef struct HashRec:
        pass

    ctypedef struct Hash:
        pass

    ctypedef struct Table:
        Uint8 *rows
        Uint64 n_bytes_per_row
        Uint64 n_max_rows
        Uint64 n_rows

    ctypedef struct DTR:
        Size count
        Index dtr_i

    ctypedef struct DyePepRec:
        Size count
        Index dtr_i
        Index pep_i

    ctypedef void (*ProgressFn)(int complete, int total, int retry)

    ctypedef struct PCB:
        Float64 pep_i
        Float64 ch_i
        Float64 p_bright

    ctypedef struct Context:
        Size n_peps
        Size n_cycles
        Size n_samples
        Size n_channels
        Uint64 pi_bleach
        Uint64 pi_detach
        Uint64 pi_edman_success
        # Annoyingly I can't get the following to reference the N_MAX_CYCLES
        # constant from above, I only seem to get this to compile by hard-coding 64!
        # I added a check for this in sanity_checks
        CycleKindType cycles[64]
        Table dtrs
        Hash dtr_hash
        Table dyepeps
        Hash dyepep_hash
        Table pcbs
        Table pep_i_to_pcb_i
        RecallType *pep_recalls
        Size count_only
        Size output_n_dtrs
        Size output_n_dyepeps
        Size n_threads
        Uint64 rng_seed
        ProgressFn progress_fn

    int setup_sanity_checks(Size n_channels, Size n_cycles)
    Uint64 prob_to_p_i(double p)
    Size dtr_n_bytes(Size n_channels, Size n_cycles)
    Table table_init(Uint8 *base, Size n_bytes, Size n_bytes_per_row)
    Table table_init_readonly(Uint8 *base, Size n_bytes, Size n_bytes_per_row)
    Hash hash_init(HashRec *buffer, Size n_max_recs)
    void context_work_orders_start(Context *ctx)
    Index context_dtr_get_count(Context *ctx, Index dtr_i)
    DyeType *context_dtr_dyetrack(Context *ctx, Index dtr_i)
    DyePepRec *context_dyepep(Context *ctx, Index dyepep_i)
    void rand64_seed(Uint64 seed)
    void context_dump(Context *ctx)
