cimport c_common as c

cdef extern from "c_sim_v2_fast.h":

    ctypedef struct HashRec:
        pass

    ctypedef struct Hash:
        pass

    ctypedef struct Dyt:
        c.Size count
        c.Index dyt_i

    ctypedef struct PCB:
        c.Float64 pep_i
        c.Float64 ch_i
        c.Float64 p_bright

    ctypedef struct DyePepRec:
        c.Index dyt_i
        c.Index pep_i
        c.Size n_reads

    ctypedef struct SimV2FastContext:
        c.Size n_peps
        c.Size n_cycles
        c.Size n_samples
        c.Size n_channels
        c.Uint64 pi_bleach
        c.Uint64 pi_detach
        c.Uint64 pi_edman_success
        # Annoyingly I can't get the following to reference the N_MAX_CYCLES
        # constant from above, I only seem to get this to compile by hard-coding 64!
        # I added a check for this in sanity_checks
        c.CycleKindType cycles[64]
        c.Tab dyts
        Hash dyt_hash
        c.Tab dyepeps
        Hash dyepep_hash
        c.Tab pcbs
        c.Tab pep_i_to_pcb_i
        c.RecallType *pep_recalls
        c.Size count_only
        c.Size output_n_dyts
        c.Size output_n_dyepeps
        c.Size n_threads
        c.Uint64 rng_seed
        c.ProgressFn progress_fn
        c.CheckKeyboardInterruptFn check_keyboard_interrupt_fn


    int setup_sanity_checks(c.Size n_channels, c.Size n_cycles)
    c.Uint64 prob_to_p_i(double p)
    c.Size dyt_n_bytes(c.Size n_channels, c.Size n_cycles)
    Hash hash_init(HashRec *buffer, c.Size n_max_recs)
    int context_work_orders_start(SimV2FastContext *ctx)
    c.Index context_dyt_get_count(SimV2FastContext *ctx, c.Index dyt_i)
    c.DyeType *context_dyt_dyetrack(SimV2FastContext *ctx, c.Index dyt_i)
    c.DyePepRec *context_dyepep(SimV2FastContext *ctx, c.Index dyepep_i)
    void rand64_seed(c.Uint64 seed)
    void context_dump(SimV2FastContext *ctx)
