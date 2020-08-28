#ifndef SIM_V2_H
#define SIM_V2_H

#include "c_common.h"


typedef struct {
    Size count;
    Index dtr_i;
    DyeType chcy_dye_counts[];
    // Note, this is a variable sized record
    // See dtr_* functions for manipulating it
} DTR;  // DTR = Dye Track Record


typedef struct {
    Float64 pep_i;
    Float64 ch_i;
    Float64 p_bright;
} PCB;  // PCB = (p)ep_i, (c)h_i, (b)right_probability


typedef struct {
    Size n_new_dtrs;
    Size n_new_dyepeps;
} Counts;


typedef struct {
    Size n_peps;
    Size n_cycles;
    Size n_samples;
    Size n_channels;
    Uint64 pi_bleach;
    Uint64 pi_detach;
    Uint64 pi_edman_success;
    CycleKindType cycles[N_MAX_CYCLES];
    Table dtrs;
    Hash dtr_hash;
    Table dyepeps;
    Hash dyepep_hash;
    Table pcbs;
    Table pep_i_to_pcb_i;
    RecallType *pep_recalls;
    Index next_pep_i;
    Size count_only;
    Size output_n_dtrs;
    Size output_n_dyepeps;
    Size n_threads;
    Uint64 rng_seed;
    pthread_mutex_t work_order_lock;
    pthread_mutex_t table_lock;
    ProgressFn progress_fn;
} SimV2FastContext;


int setup_sanity_checks(Size n_channels, Size n_cycles);
Uint64 prob_to_p_i(double p);
void rand64_seed(Uint64 seed);
Size dtr_n_bytes(Size n_channels, Size n_cycles);
void context_work_orders_start(SimV2FastContext *ctx);
Index context_dtr_get_count(SimV2FastContext *ctx, Index dtr_i);
DyeType *context_dtr_dyetrack(SimV2FastContext *ctx, Index dtr_i);
DyePepRec *context_dyepep(SimV2FastContext *ctx, Index dyepep_i);
void context_dump(SimV2FastContext *ctx);

#endif
