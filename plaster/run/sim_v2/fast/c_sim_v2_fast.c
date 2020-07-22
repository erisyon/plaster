#include "math.h"
#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "inttypes.h"
#include "c_sim_v2_fast.h"

/*
This is the "sim" phase of plaster implemented in C.
It is meant to be called by Cython sim_v2_fast.pyx

Inputs (see typedef Context in sim.h):
    * A list of "flus" which are Uint8 arrays (n_channels, n_cycles)
      with the channel number of NO_LABEL at each position.
    * Various paramerters
    * Working buffers
    * All memory is allocated/freed/managed by the caller

Algorithm:
    For each peptide "flu":
        Allocate a buffer for a working copy of the flu
        For n_samples Monte-Carlo simulations:
            * Copy the master peptide flu into the working copy
            * For each chemical cycle:
                  * Remove the leading edge aa (or edman fail)
                  * Allow each fluophore on the working copy an
                    opportunity to bleach
                  * Allow the flu an opportunity to "detach" (all fluors go dark)
                  * "Image" by suming up the remaining dyes.
            * We now have a dyetrack
            * Make a 64 bit hash key from that dyetrack
            * Look up the dyetrack in the Dye Track Record (DTR) Hash;
              if it has never been seen before, add it; increment count.
            * Make another 64 bit hash key by combining the dyetrack hash key
              with the pep_i.
            * Lookup using this "DyePep" hash key into the Dye Pep Hash;
              if it has never been seen before, add it; increment count.

Definitions:
    DTR = Dye Track Record
    DyePepRec = A record that associates (dye_i, pep_i, count)
    Table = A generic object that tracks how many rows have been added
        into a growing array. The pre-allocated table buffer must large
        enough to accommodate the row or an assertion will be thrown.
    Hash = A simple 64-bit hashkey table that maintains a pointer value.
    Context = All of the context (parameters, buffers, inputs, etc)
        that are needed in order to run the simulation

There are two Tables maintained in the context:
    dtrs: (count, dtr_i, array(n_channels, n_cycles))
    dyepeps: (count, dtr_i, pep_i)

There are two hash tables:
    dtr_hash: key=dyetrack (note: not dtr_i), val=(count, dtr_i)
    dyepep_hash: key=(dyetrack, pep_i) , val=(count, dtr_i, pep_i)
*/


// Helpers
//=========================================================================================

// See setup() and *_get_haskey()
static Uint64 hashkey_factors[256];


static Uint128 rng_state = 1;
void rand64_seed(Uint64 seed) {
    rng_state = seed;
}

int rand64(Uint64 p_i) {
    // p_i is a unsigned 64-bit probability.
    // When p_i is small this function is likely to return 0
    // TODO: Consider a better RNG here
    rng_state *= (Uint128)0xda942042e4dd58b5;
    return (rng_state >> 64) < p_i ? 1 : 0;
}


Uint64 prob_to_p_i(double p) {
    // Convert p (double 0-1) into a 64 bit integer
    ensure(0.0 <= p && p <= 1.0, "probability out of range");
    long double w = floorl( (long double)p * (long double)(UINT64_MAX) );
    Uint64 ret = (Uint64)w;
    // printf("ret=%" PRIu64 "\n", ret);
    return ret;
}


int setup_and_sanity_check(Size n_channels, Size n_cycles) {
    // Setup the hashkey_factors with random numbers and
    // Check that the compiler sizes are what is expected.
    // return 0 == success

    if(UINT64_MAX != 0xFFFFFFFFFFFFFFFFULL) {
        printf("Failed sanity check: UINT64_MAX\n");
        return 1;
    }

    if(sizeof(Uint128) != 16) {
        printf("Failed sanity check: sizeof 128\n");
        return 2;
    }

    if(sizeof(Uint64) != 8) {
        printf("Failed sanity check: sizeof 64\n");
        return 3;
    }

    if(sizeof(Uint8) != 1) {
        printf("Failed sanity check: sizeof 8\n");
        return 4;
    }

    if(sizeof(Sint64) != 8) {
        printf("Failed sanity check: sizeof Sint64\n");
        return 5;
    }

    if(rand64(UINT64_MAX) != 1) {
        printf("Failed sanity check: rand64\n");
        return 6;
    }

    if(sizeof(DTR) != 16) {
        printf("Failed sanity check: DTR size\n");
        return 7;
    }

    if(N_MAX_CYCLES != 64) {
        // This is particularly annoying. See csim.pxd for explanation
        printf("Failed sanity check: N_MAX_CYCLES\n");
        return 8;
    }

    Size n_hashkey_factors = sizeof(hashkey_factors) / sizeof(hashkey_factors[0]);
    for(Index i=0; i<n_hashkey_factors; i++) {
        hashkey_factors[i] = (rand() * rand() * rand() * rand() * rand() * rand() * rand()) % UINT64_MAX;
    }

    if(n_channels * n_cycles >= n_hashkey_factors) {
        return 9;
    }

    if(prob_to_p_i(0.0) != (Uint64)0) {
        printf("Failed sanity check: prob_to_p_i(0.0)\n");
        return 10;
    }

    if(prob_to_p_i(1.0) != (Uint64)UINT64_MAX) {
        printf("Failed sanity check: prob_to_p_i(1.0) %ld %ld\n", prob_to_p_i(1.0), (Uint64)UINT64_MAX);
        return 11;
    }

    if(sizeof(Float64) != 8) {
        printf("Failed sanity check: Float64\n");
        return 12;
    }

    return 0;
}


// Hash
//=========================================================================================

Hash hash_init(HashRec *buffer, Size n_max_recs) {
    Hash hash;
    hash.recs = (HashRec *)buffer;
    memset(buffer, 0, n_max_recs * sizeof(HashRec));
    hash.n_max_recs = n_max_recs;
    hash.n_active_recs = 0;
    return hash;
}


HashRec *hash_get(Hash hash, HashKey key) {
    /*
    Assumes the hash table is large enough to never over-flow.

    Usage:
        Hash hash = hash_init(buffer, n_recs);
        HashRec *rec = hash_get(hash, key);
        if(rec == (HashRec*)0) {
            // hash full!
        }
        else if(rec->key == 0) {
            // New record
        }
        else {
            // Existing record
        }
    */
    ensure(key != 0, "Invalid hashkey");
    Index i = key % hash.n_max_recs;
    Index start_i = i;
    HashKey key_at_i = hash.recs[i].key;
    while(key_at_i != key) {
        if(key_at_i == 0) {
            return &hash.recs[i];
        }
        i = (i + 1) % hash.n_max_recs;
        key_at_i = hash.recs[i].key;
        if(i == start_i) {
            return (HashRec *)0;
        }
    }
    return &hash.recs[i];
}


void hash_dump(Hash hash) {
    // Debugging
    for(Index i=0; i<hash.n_max_recs; i++) {
        printf("%08ld: %016lX %p\n", i, hash.recs[i].key, hash.recs[i].val);
    }
}


// dtrs = Dye Track Records
//=========================================================================================

HashKey dtr_get_hashkey(DTR *dtr, Size n_channels, Size n_cycles) {
    // Get a hashkey for the DTR by a dot product with a set of random 64-bit
    // values initialized in the hashkey_factors
    HashKey key = 0;
    Uint64 *p = hashkey_factors;
    DyeType *d = dtr->chcy_dye_counts;
    for(Index i=0; i < n_channels * n_cycles; i++) {
        key += (*p++) * (Uint64)(*d++);
    }
    return key + 1;  // +1 to reserve 0
}


Size dtr_n_bytes(Size n_channels, Size n_cycles) {
    // Return aligned DTR size
    Size size = sizeof(DTR) + sizeof(DyeType) * n_cycles * n_channels;
    int over = size % 8;
    int padding = over == 0 ? 0 : 8 - over;
    return size + padding;
}


void dtr_set_chcy(DTR *dst, DyeType src_val, Size n_channels, Size n_cycles, Index ch_i, Index cy_i) {
    // DTR chcy_dye_counts is a 2D array (n_channels, n_cycles)
    ensure_only_in_debug(0 <= ch_i && ch_i < n_channels && 0 <= cy_i && cy_i < n_cycles, "dtr set out of bounds");
    Uint64 index = (n_cycles * ch_i) + cy_i;
    ensure_only_in_debug(0 <= index && index < n_channels * n_cycles, "dtr set out of bounds index");
    dst->chcy_dye_counts[index] = src_val;
}


void dtr_clear(DTR *dst, Size n_channels, Size n_cycles) {
    // Clear a single DTR
    memset(dst->chcy_dye_counts, 0, sizeof(DyeType) * n_channels * n_cycles);
}


Size dtr_sum(DTR *dtr, Size n_chcy) {
    // return the sum of all channel, all cycles (for debugging)
    Size sum = 0;
    for(Index i=0; i<n_chcy; i++) {
        sum += dtr->chcy_dye_counts[i];
    }
    return sum;
}


void dtr_dump_one(DTR *dtr, Size n_channels, Size n_cycles) {
    // debugging
    for(Index ch_i=0; ch_i<n_channels; ch_i++) {
        for(Index cy_i=0; cy_i<n_cycles; cy_i++) {
            printf("%d ", dtr->chcy_dye_counts[ch_i*n_cycles + cy_i]);
        }
        printf("  ");
    }
    printf(": count=%4ld\n", dtr->count);
}


void dtr_dump_all(Table *dtrs, Size n_channels, Size n_cycles) {
    // debugging
    for(Index i=0; i<dtrs->n_rows; i++) {
        DTR *dtr = table_get_row(dtrs, i, DTR);
        dtr_dump_one(dtr, n_channels, n_cycles);
    }
}

void dtr_dump_one_hex(DTR *dtrs, Size n_dtrs, Size n_channels, Size n_cycles) {
    // debugging
    DTR *rec = dtrs;
    Uint8 *ptr = (Uint8 *)dtrs;
    for(Index i=0; i<n_dtrs; i++) {
        HashKey key = dtr_get_hashkey(rec, n_channels, n_cycles);
        printf("%016lX ", key);
        for(Index i=0; i<8; i++) {
            printf("%02x", *ptr++);
        }
        printf("  ");
        for(Index ch_i=0; ch_i<n_channels; ch_i++) {
            for(Index cy_i=0; cy_i<n_cycles; cy_i++) {
                printf("%02x ", *ptr++);
            }
            printf("  ");
        }
        printf("\n");
        rec++;
    }
}


// DyePep
//=========================================================================================

HashKey dyepep_get_hashkey(HashKey dtr_hashkey, Index pep_i) {
    // Note, 0 is an illegal return but is very unlikely except
    // under very weird circumstances. The check is therefore only
    // performec under DEBUG
    HashKey key = dtr_hashkey * hashkey_factors[0] + pep_i * hashkey_factors[1] + 1;  // + 1 to reserve 0
    ensure_only_in_debug(key != 0, "dyepep hash == 0");
    return key;
}


void dyepep_dump_one(DyePepRec *dyepep) {
    // Debugging
    printf("%4ld %4ld %4ld\n", dyepep->dtr_i, dyepep->pep_i, dyepep->count);
}


void dyepep_dump_all(Table *dyepeps) {
    // Debugging
    for(Index i=0; i<dyepeps->n_rows; i++) {
        dyepep_dump_one(table_get_row(dyepeps, i, DyePepRec));
    }
}


// sim
//=========================================================================================

void context_sim_flu(Context *ctx, Index pep_i) {
    // Runs the Monte-Carlo simulation of one peptide flu over n_samples
    // See algorithm described at top of file.

    // Make local copies of inner-loop variables
    DyeType ch_sums[N_MAX_CHANNELS];
    Size n_cycles = ctx->n_cycles;
    Size n_samples = ctx->n_samples;
    Size n_channels = ctx->n_channels;
    DyeType *flu = ctx->flus[pep_i];
    PIType *pi_bright = ctx->pi_brights[pep_i];
    Size n_aas = ctx->n_aas[pep_i];
    CycleKindType *cycles = ctx->cycles;
    Uint64 pi_bleach = ctx->pi_bleach;
    Uint64 pi_detach = ctx->pi_detach;
    Uint64 pi_edman_success = ctx->pi_edman_success;
    Table *dtrs = &ctx->dtrs;
    Table *dyepeps = &ctx->dyepeps;
    Hash dtr_hash = ctx->dtr_hash;
    Hash dyepep_hash = ctx->dyepep_hash;

    // working_flu is volatile stack copy of the incoming flu
    Size n_working_flu_bytes = sizeof(DyeType) * n_aas;
    DyeType *working_flu = (DyeType *)alloca(n_working_flu_bytes);
    memset(working_flu, 0, n_working_flu_bytes);

    // working_dtr is volatile stack copy of the out-going DTR
    Size n_dyetrack_bytes = dtr_n_bytes(ctx->n_channels, ctx->n_cycles);
    DTR *working_dtr = (DTR *)alloca(n_dyetrack_bytes);
    memset(working_dtr, 0, n_dyetrack_bytes);

    DTR *nul_dtr = (DTR *)alloca(n_dyetrack_bytes);
    memset(nul_dtr, 0, n_dyetrack_bytes);

    Size n_dark_samples = 0;
    Size n_non_dark_samples = 0;
    while(n_non_dark_samples < n_samples) {
        if(n_dark_samples > 10 * n_samples) {
            // Emergency exit. The recall is so low that we need to
            // just give up and declare that it can't be measured.
            n_dark_samples = 0;
            n_non_dark_samples = 0;
            break;
        }

        // GENERATE the working_dyetrack sample (Monte Carlo)
        //-------------------------------------------------------
        memcpy(working_flu, flu, n_working_flu_bytes);
        dtr_clear(working_dtr, n_channels, n_cycles);

        // MODEL dark-dyes (dyes dark before the first image)
        // These darks are the product of various dye factors which
        // are passed into this module already converted into PI form
        // (probability in 0 - max_unit64) by the pi_bright arrays
        for(Index aa_i=0; aa_i<n_aas; aa_i++) {
            if( ! rand64(pi_bright[aa_i])) {
                working_flu[aa_i] = NO_LABEL;
            }
        }

        Index head_i = 0;
        for(Index cy_i=0; cy_i<n_cycles; cy_i++) {
            // EDMAN...
            // Edman degrdation chews off the N-terminal amino-acid.
            // If successful this advances the "head_i" which is where we're summing from.
            if(cycles[cy_i] == CYCLE_TYPE_EDMAN) {
                if(rand64(pi_edman_success)) {
                    head_i ++;
                }
            }

            // DETACH...
            // Detachment is when a peptide comes loose from the surface.
            // This means that all subsequent measurements go dark.
            if(rand64(pi_detach)) {
                for(Index aa_i=head_i; aa_i<n_aas; aa_i++) {
                    working_flu[aa_i] = NO_LABEL;
                }
                break;
            }

            // IMAGE (sum up all active dyes in each channel)...
            // To make this avoid any branching logic, the ch_sums[]
            // is allocated to with N_MAX_CHANNELS which includes the "NO_LABEL"
            // which is defined to be N_MAX_CHANNELS-1. Thus the sums
            // will also count the number of unlabelled positions, but
            // we can just ignore that extra "NO LABEL" channel.
            memset(ch_sums, 0, sizeof(ch_sums));
            for(Index aa_i=head_i; aa_i<n_aas; aa_i++) {
                ch_sums[working_flu[aa_i]] ++;
            }
            for(Index ch_i=0; ch_i<n_channels; ch_i++) {
                dtr_set_chcy(working_dtr, ch_sums[ch_i], n_channels, n_cycles, ch_i, cy_i);
            }

            // TODO: Explain why BLEACH is treated POST image
            //   Ie only makes a difference to the first cycle
            // BLEACH
            for(Index aa_i=head_i; aa_i<n_aas; aa_i++) {
                // For all REMAINING dyes (head_i:...) give
                // each dye a chance to photobleach.
                // TODO: Profile which is better, the branch here or just letting it over-write
                if(working_flu[aa_i] < NO_LABEL && rand64(pi_bleach)) {
                    working_flu[aa_i] = NO_LABEL;
                }
            }
        }

        // At this point we have the flu sampled into working_dtr
        // Now we look it up in the hash tables.
        //-------------------------------------------------------

        if(memcmp(working_dtr, nul_dtr, n_dyetrack_bytes) == 0) {
            // The row was empty, note this and continue to try another sample
            n_dark_samples++;
            continue;
        }

        n_non_dark_samples++;

        HashKey dtr_hashkey = dtr_get_hashkey(working_dtr, n_channels, n_cycles);
        HashRec *dtr_hash_rec = hash_get(dtr_hash, dtr_hashkey);
        DTR *dtr;
        ensure(dtr_hash_rec != (HashRec*)0, "dtr_hash full");
        if(dtr_hash_rec->key == 0) {
            // New record
            Index dtr_i = table_add(dtrs, working_dtr, ctx->n_threads > 1 ? &ctx->table_lock : 0);
            dtr = table_get_row(dtrs, dtr_i, DTR);
            dtr_hash_rec->key = dtr_hashkey;
            dtr->count++;
            dtr->dtr_i = dtr_i;
            dtr_hash_rec->val = dtr;
        }
        else {
            // Existing record
            // Because this is a MonteCarlo sampling it really doesn't
            // matter if we occasionally mis-count due to thread
            // contention therefore there is no lock here.
            dtr = (DTR *)(dtr_hash_rec->val);
            table_validate_only_in_debug(dtrs, dtr, "after val lookup");
            dtr->count++;
        }
        table_validate_only_in_debug(dtrs, dtr, "after dtr setup");

        // SAVE the (dtr_i, pep_i) into dyepeps
        // (or inc count if it already exists)
        //-------------------------------------------------------
        HashKey dyepep_hashkey = dyepep_get_hashkey(dtr_hashkey, pep_i);
        HashRec *dyepep_hash_rec = hash_get(dyepep_hash, dyepep_hashkey);
        ensure(dyepep_hash_rec != (HashRec*)0, "dyepep_hash full");
        if(dyepep_hash_rec->key == 0) {
            // New record
            // If this were used multi-threaded, this would be a race condition
            Index dyepep_i = table_add(dyepeps, 0, ctx->n_threads > 1 ? &ctx->table_lock : 0);
            DyePepRec *dyepep = table_get_row(dyepeps, dyepep_i, DyePepRec);
            dyepep_hash_rec->key = dyepep_hashkey;
            dyepep->dtr_i = dtr->dtr_i;
            dyepep->pep_i = pep_i;
            dyepep->count++;
            dyepep_hash_rec->val = dyepep;
        }
        else {
            // Existing record
            // Same argument as above
            DyePepRec *dpr = (DyePepRec *)dyepep_hash_rec->val;
            table_validate_only_in_debug(dyepeps, dpr, "dyepep hash inc");
            dpr->count++;
        }
    }

    if(n_dark_samples + n_non_dark_samples > 0) {
        ctx->pep_recalls[pep_i] = (double)n_non_dark_samples / (double)(n_dark_samples + n_non_dark_samples);
    }
    else {
        ctx->pep_recalls[pep_i] = (double)0.0;
    }
}


void _context_generate_test_pepflus(Context *ctx) {
    // Mock flues for testing purposes
    ctx->flus = (DyeType **)calloc(sizeof(DyeType *), ctx->n_peps);
    ctx->n_aas = (Size *)calloc(sizeof(Size), ctx->n_peps);
    Size n_channels = ctx->n_channels;
    for(Index pep_i=0; pep_i<ctx->n_peps; pep_i++) {
        Size n_aa = 5 + rand() % 20;
        ctx->n_aas[pep_i] = n_aa;
        ctx->flus[pep_i] = (DyeType *)calloc(sizeof(DyeType), n_aa);
        for(Index i=0; i<n_aa; i++) {
            ctx->flus[pep_i][i] = NO_LABEL;
            for(Index ch_i=(rand() % n_channels); ch_i<n_channels; ch_i++) {
                if(!(rand() % 4)) {
                    ctx->flus[pep_i][i] = (DyeType)(ch_i % n_channels);
                    break;
                }
            }
        }
    }
}


Index context_work_orders_pop(Context *ctx) {
    // NOTE: This return +1! So that 0 can be reserved.
    if(ctx->n_threads > 1) {
        pthread_mutex_lock(&ctx->work_order_lock);
    }

    Index i = ctx->next_pep_i;
    ctx->next_pep_i++;

    if(ctx->n_threads > 1) {
        pthread_mutex_unlock(&ctx->work_order_lock);
    }

    if(i < ctx->n_peps) {
        return i + 1;
    }
    return 0;
}


void *context_work_orders_worker(void *_ctx) {
    // The worker thread. Pops off which pep to work on next
    // continues until there are no more work orders.
    Context *ctx = (Context *)_ctx;
    while(1) {
        Index pep_i_plus_1 = context_work_orders_pop(ctx);
        if(pep_i_plus_1 == 0) {
            break;
        }
        Index pep_i = pep_i_plus_1 - 1;
        context_sim_flu(ctx, pep_i);
    }
    return (void *)0;
}


void context_work_orders_start(Context *ctx) {
    // context_dump(ctx);

    // Initialize mutex and start the worker thread(s).
    ensure(setup_and_sanity_check(ctx->n_channels, ctx->n_cycles) == 0, "Sanity checks failed");
    rand64_seed(ctx->rng_seed);

    ctx->next_pep_i = 0;

    // Add a nul-row
    Size n_dyetrack_bytes = dtr_n_bytes(ctx->n_channels, ctx->n_cycles);
    DTR *nul_rec = (DTR *)alloca(n_dyetrack_bytes);
    memset(nul_rec, 0, n_dyetrack_bytes);
    HashKey dtr_hashkey = dtr_get_hashkey(nul_rec, ctx->n_channels, ctx->n_cycles);
    HashRec *dtr_hash_rec = hash_get(ctx->dtr_hash, dtr_hashkey);
    ensure(dtr_hash_rec->key == 0, "dtr hash should not have found nul row");

    Table *dtrs = &ctx->dtrs;
    Index nul_i = table_add(dtrs, nul_rec, (void*)0);
    DTR *nul_dtr = table_get_row(dtrs, nul_i, DTR);
    dtr_hash_rec->key = dtr_hashkey;
    nul_dtr->count++;
    nul_dtr->dtr_i = nul_i;
    dtr_hash_rec->val = nul_dtr;

    pthread_t ids[256];
    ensure(0 < ctx->n_threads && ctx->n_threads < 256, "Invalid n_threads");

    if(ctx->n_threads > 1) {
        int ret = pthread_mutex_init(&ctx->work_order_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");

        ret = pthread_mutex_init(&ctx->table_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        int ret = pthread_create(&ids[i], NULL, context_work_orders_worker, ctx);
        ensure(ret == 0, "Thread not created.");
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        pthread_join(ids[i], NULL);
    }

    // trace("dtrs n_rows = %ld\n", ctx->dtrs.n_rows);
}


Index context_dtr_get_count(Context *ctx, Index dtr_i) {
    DTR *dtr = table_get_row(&ctx->dtrs, dtr_i, DTR);
    return dtr->count;
}


DyeType *context_dtr_dyetrack(Context *ctx, Index dtr_i) {
    DTR *dtr = table_get_row(&ctx->dtrs, dtr_i, DTR);
    return dtr->chcy_dye_counts;
}


DyePepRec *context_dyepep(Context *ctx, Index dyepep_i) {
    return table_get_row(&ctx->dyepeps, dyepep_i, DyePepRec);
}


void context_dump(Context *ctx) {
    printf("Context:\n");
    printf("  n_peps=%" PRIu64 "\n", ctx->n_peps);
    printf("  n_cycles=%" PRIu64 "\n", ctx->n_cycles);
    printf("  n_samples=%" PRIu64 "\n", ctx->n_samples);
    printf("  n_channels=%" PRIu64 "\n", ctx->n_channels);
        // printf("ret=%" PRIu64 "\n", ret);

    printf("  pi_bleach=%" PRIu64 "\n", ctx->pi_bleach);
    printf("  pi_detach=%" PRIu64 "\n", ctx->pi_detach);
    printf("  pi_edman_success=%" PRIu64 "\n", ctx->pi_edman_success);
    printf("  n_threads=%" PRIu64 "\n", ctx->n_threads);
    printf("  rng_seed=%" PRIu64 "\n", ctx->rng_seed);
    // Some are left out
}


/*
int main() {
    // Tests (not run by production code, see fast_sim.pyx)
    // Setup context
    Context ctx;

    ctx.n_peps = 100;
    ctx.n_samples = 5000;
    ctx.n_channels = 3;
    ctx.n_cycles = 15;

    if(setup_and_sanity_check(ctx.n_channels, ctx.n_cycles) != 0) {
        return 1;
    }

    ctx.pi_bleach = prob_to_p_i(0.06);
    ctx.pi_detach = prob_to_p_i(0.04);
    ctx.pi_edman_success = prob_to_p_i(1.0 - 0.05);
    ctx.cycles[0] = CYCLE_TYPE_PRE;
    for(Index i=0; i<ctx.n_cycles; i++) {
        ctx.cycles[i] = i==0 ? CYCLE_TYPE_PRE : CYCLE_TYPE_EDMAN;
    }
    _context_generate_test_pepflus(&ctx);

    Size n_trials = 20;
    //Uint64 start = now();
    for(Index i=0; i<n_trials; i++) {
        Size n_max_dtrs = 8 * ctx.n_peps * ctx.n_samples / 10;
        Size n_dtr_row_bytes = dtr_n_bytes(ctx.n_channels, ctx.n_cycles);
        ctx.dtrs = table_init(calloc(n_max_dtrs, n_dtr_row_bytes), n_max_dtrs * n_dtr_row_bytes, n_dtr_row_bytes);

        Size n_max_dtr_hash_recs = 2 * n_max_dtrs;
        HashRec *dtr_hash_buffer = (HashRec *)calloc(n_max_dtr_hash_recs, sizeof(HashRec));
        ctx.dtr_hash = hash_init(dtr_hash_buffer, n_max_dtr_hash_recs);

        Size n_max_dyepeps = 3 * n_max_dtrs;
        ctx.dyepeps = table_init(calloc(n_max_dyepeps, sizeof(DyePepRec)), n_max_dyepeps * sizeof(DyePepRec), sizeof(DyePepRec));

        Size n_max_dyepep_hash_recs = 2 * n_max_dyepeps;
        HashRec *dyepep_hash_buffer = (HashRec *)calloc(n_max_dyepep_hash_recs, sizeof(HashRec));
        ctx.dyepep_hash = hash_init(dyepep_hash_buffer, n_max_dyepep_hash_recs);

        ctx.next_pep_i = 0;

        ctx.n_threads = 2;
        context_work_orders_start(&ctx);
    }
    //Uint64 stop = now();
    //trace("dtrs n_rows = %ld\n", ctx.dtrs.n_rows);
    //trace("%f sec per trial\n", (double)(stop-start) / ((double)n_trials * 1000.0*1000.0) );
    return 0;
}
*/