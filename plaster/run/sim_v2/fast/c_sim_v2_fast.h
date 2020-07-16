#ifndef SIM_V2_H
#define SIM_V2_H

// See sim.c for docs.

typedef __uint8_t Uint8;
typedef __uint64_t Uint64;
typedef __uint128_t Uint128;
typedef __int128_t Sint128;
typedef __int64_t Sint64;
typedef double Float64;


typedef Uint64 Size;
typedef Uint64 Index;
typedef Uint64 HashKey;
typedef Uint8 DyeType;
typedef Uint8 CycleKindType;
typedef Uint64 PIType;
typedef Float64 RecallType;


#define N_MAX_CHANNELS ((DyeType)(8))
#define NO_LABEL ((DyeType)(N_MAX_CHANNELS - 1))
#define N_MAX_CYCLES ((DyeType)64)
#define CYCLE_TYPE_PRE ((CycleKindType)(0))
#define CYCLE_TYPE_MOCK ((CycleKindType)(1))
#define CYCLE_TYPE_EDMAN ((CycleKindType)(2))


typedef struct {
    HashKey key;
    void *val;
} HashRec;


typedef struct {
    HashRec *recs;
    Uint64 n_max_recs;
    Uint64 n_active_recs;
} Hash;


typedef struct {
    Uint8 *rows;
    Uint64 n_bytes_per_row;
    Uint64 n_max_rows;
    Uint64 n_rows;
} Table;


typedef struct {
    Size count;
    Index dtr_i;
    DyeType chcy_dye_counts[];
    // Note, this is a variable sized record
    // See dtr_* functions for manipulating it
} DTR;  // DTR = Dye Track Record


typedef struct {
    Size count;
    Index dtr_i;
    Index pep_i;
} DyePepRec;


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
    DyeType **flus;
    PIType **pi_brights;
    Size *n_aas;
    RecallType *pep_recalls;
    Index next_pep_i;
    Size n_threads;
    Uint64 rng_seed;
    pthread_mutex_t work_order_lock;
    pthread_mutex_t table_lock;
} Context;


int setup_sanity_checks(Size n_channels, Size n_cycles);
Uint64 prob_to_p_i(double p);
void rand64_seed(Uint64 seed);
Size dtr_n_bytes(Size n_channels, Size n_cycles);
Table table_init(Uint8 *base, Size n_bytes, Size n_bytes_per_row);
Hash hash_init(HashRec *buffer, Size n_max_recs);
void context_work_orders_start(Context *ctx);
Index context_dtr_get_count(Context *ctx, Index dtr_i);
DyeType *context_dtr_dyetrack(Context *ctx, Index dtr_i);
DyePepRec *context_dyepep(Context *ctx, Index dyepep_i);
void context_dump(Context *ctx);

#endif
