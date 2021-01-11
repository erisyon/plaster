#include "alloca.h"
#include "inttypes.h"
#include "math.h"
#include "memory.h"
#include "pthread.h"
#include "stdarg.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"

#include "c_common.h"

#include "_sim_v2.h"

/*
This is the "sim" phase of plaster implemented in C.
It is meant to be called by Cython sim_v2_fast.pyx

Inputs (see typedef SimV2Context in sim.h):
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
            * Look up the dyetrack in the Dye Tracks (Dyts) Hash;
              if it has never been seen before, add it; increment count.
            * Make another 64 bit hash key by combining the dyetrack hash key
              with the pep_i.
            * Lookup using this "DyePep" hash key into the Dye Pep Hash;
              if it has never been seen before, add it; increment count.

Definitions:
    Dyt = Dye Track - a monotonically decreasing array of dyecounts ex: 3, 3, 2, 2, 2, 1, 0
    DyePepRec = A record that associates (dye_i, pep_i, count)
    Tab = A generic object that tracks how many rows have been added
        into a growing array. The pre-allocated tab buffer must large
        enough to accommodate the row or an assertion will be thrown.
    Hash = A simple 64-bit hashkey tab that maintains a pointer value.
    SimV2Context = All of the context (parameters, buffers, inputs, etc)
        that are needed in order to run the simulation
    Flu = Fluoro label sequence -
        For peptide: ABCDEF
        flu:         .10.01
        Labels: ch0 labels CE, c1 labels BF
        the encoding of "." is max_channels-1
    p_* = floating point probability of * (0-1)
    pi_* = integer probability where (0-1) is mapped to (0-MAX_UINT)
    bright_prob = the inverse of all the ways a dye can fail to be visible.
        In other words, the probability that a dye is active, ie bright


There are two Tabs maintained in the context:
    dyts: (count, dyt_i, array(n_channels, n_cycles))
    dyepeps: (count, dyt_i, pep_i)

There are two hash tabs:
    dyt_hash: key=dyetrack (note: not dyt_i), val=(count, dyt_i)
    dyepep_hash: key=(dyetrack, pep_i) , val=(count, dyt_i, pep_i)
*/

// Helpers
//=========================================================================================

// See setup() and *_get_haskey()
static Uint64 hashkey_factors[256];

static Uint128 rng_state = 1;
void rand64_seed(Uint64 seed) { rng_state = seed; }

int rand64(Uint64 p_i) {
    // p_i is a unsigned 64-bit probability.
    // When p_i is small this function is likely to return 0
    // TODO: Consider a better RNG here
    rng_state *= (Uint128)0xda942042e4dd58b5;
    return (rng_state >> 64) < p_i ? 1 : 0;
}

PIType prob_to_p_i(double p) {
    // Convert p (double 0-1) into a 64 bit integer
    ensure(0.0 <= p && p <= 1.0, "probability out of range");
    long double w = floorl((long double)p * (long double)(UINT64_MAX));
    Uint64 ret = (Uint64)w;
    // printf("ret=%" PRIu64 "\n", ret);
    return ret;
}

int setup_and_sanity_check(Size n_channels, Size n_cycles) {
    // Setup the hashkey_factors with random numbers and
    // Check that the compiler sizes are what is expected.
    // return 0 == success

    if(rand64(UINT64_MAX) != 1) {
        printf("Failed sanity check: rand64\n");
        return 6;
    }

    if(sizeof(Dyt) != 16) {
        printf("Failed sanity check: Dyt size\n");
        return 7;
    }

    Size n_hashkey_factors = sizeof(hashkey_factors) / sizeof(hashkey_factors[0]);
    for(Index i = 0; i < n_hashkey_factors; i++) {
        hashkey_factors[i] = (rand() * rand() * rand() * rand() * rand() * rand() * rand()) % UINT64_MAX;
    }

    if(n_channels * n_cycles >= n_hashkey_factors) {
        printf("Failed sanity check: n_channels * n_cycles >= n_hashkey_factors\n");
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

    return 0;
}

// Dyts = Dye tracks
//=========================================================================================

HashKey dyt_get_hashkey(Dyt *dyt, Size n_channels, Size n_cycles) {
    // Get a hashkey for the Dyt by a dot product with a set of random 64-bit
    // values initialized in the hashkey_factors
    HashKey key = 0;
    Uint64 *p = hashkey_factors;
    DyeType *d = dyt->chcy_dye_counts;
    for(Index i = 0; i < n_channels * n_cycles; i++) {
        key += (*p++) * (Uint64)(*d++);
    }
    return key + 1; // +1 to reserve 0
}

Size dyt_n_bytes(Size n_channels, Size n_cycles) {
    // Return aligned Dyt size
    Size size = sizeof(Dyt) + sizeof(DyeType) * n_cycles * n_channels;
    int over = size % 8;
    int padding = over == 0 ? 0 : 8 - over;
    return size + padding;
}

void dyt_set_chcy(Dyt *dst, DyeType src_val, Size n_channels, Size n_cycles, Index ch_i, Index cy_i) {
    // Dyt chcy_dye_counts is a 2D array (n_channels, n_cycles)
    ensure_only_in_debug(0 <= ch_i && ch_i < n_channels && 0 <= cy_i && cy_i < n_cycles, "dyt set out of bounds");
    Uint64 index = (n_cycles * ch_i) + cy_i;
    ensure_only_in_debug(0 <= index && index < n_channels * n_cycles, "dyt set out of bounds index");
    dst->chcy_dye_counts[index] = src_val;
}

void dyt_clear(Dyt *dst, Size n_channels, Size n_cycles) {
    // Clear a single Dyt
    memset(dst->chcy_dye_counts, 0, sizeof(DyeType) * n_channels * n_cycles);
}

Size dyt_sum(Dyt *dyt, Size n_chcy) {
    // return the sum of all channel, all cycles (for debugging)
    Size sum = 0;
    for(Index i = 0; i < n_chcy; i++) {
        sum += dyt->chcy_dye_counts[i];
    }
    return sum;
}

void dyt_dump_one(Dyt *dyt, Size n_channels, Size n_cycles) {
    // debugging
    for(Index ch_i = 0; ch_i < n_channels; ch_i++) {
        for(Index cy_i = 0; cy_i < n_cycles; cy_i++) {
            printf("%d ", dyt->chcy_dye_counts[ch_i * n_cycles + cy_i]);
        }
        printf("  ");
    }
    printf(": count=%4ld\n", dyt->count);
}

void dyt_dump_all(Tab *dyts, Size n_channels, Size n_cycles) {
    // debugging
    for(Index i = 0; i < dyts->n_rows; i++) {
        tab_var(Dyt, dyt, dyts, i);
        dyt_dump_one(dyt, n_channels, n_cycles);
    }
}

void dyt_dump_one_hex(Dyt *dyts, Size n_dyts, Size n_channels, Size n_cycles) {
    // debugging
    Dyt *rec = dyts;
    Uint8 *ptr = (Uint8 *)dyts;
    for(Index i = 0; i < n_dyts; i++) {
        HashKey key = dyt_get_hashkey(rec, n_channels, n_cycles);
        printf("%016lX ", key);
        for(Index i = 0; i < 8; i++) {
            printf("%02x", *ptr++);
        }
        printf("  ");
        for(Index ch_i = 0; ch_i < n_channels; ch_i++) {
            for(Index cy_i = 0; cy_i < n_cycles; cy_i++) {
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

HashKey dyepep_get_hashkey(HashKey dyt_hashkey, Index pep_i) {
    // Note, 0 is an illegal return but is very unlikely except
    // under very weird circumstances. The check is therefore only
    // performec under DEBUG
    HashKey key = dyt_hashkey * hashkey_factors[0] + pep_i * hashkey_factors[1] + 1; // + 1 to reserve 0
    ensure_only_in_debug(key != 0, "dyepep hash == 0");
    return key;
}

void dyepep_dump_one(DyePepRec *dyepep) {
    // Debugging
    printf("%4ld %4ld %4ld\n", dyepep->dyt_i, dyepep->pep_i, dyepep->n_reads);
}

void dyepep_dump_all(Tab *dyepeps) {
    // Debugging
    for(Index i = 0; i < dyepeps->n_rows; i++) {
        dyepep_dump_one(tab_ptr(DyePepRec, dyepeps, i));
    }
}

// sim
//=========================================================================================

Counts context_sim_flu(SimV2Context *ctx, Index pep_i, Tab *pcb_block, Size n_aas) {
    // Runs the Monte-Carlo simulation of one peptide flu over n_samples
    // See algorithm described at top of file.
    // Returns the number of NEW dyts

    // Make local copies of inner-loop variables
    DyeType ch_sums[N_MAX_CHANNELS];
    Size n_cycles = ctx->n_cycles;
    Size n_samples = ctx->n_samples;
    Size n_channels = ctx->n_channels;
    CycleKindType *cycles = ctx->cycles;
    Uint64 pi_bleach = ctx->pi_bleach;
    Uint64 pi_detach = ctx->pi_detach;
    Uint64 pi_edman_success = ctx->pi_edman_success;
    Uint64 prevent_edman_cterm = ctx->prevent_edman_cterm;
    Tab *dyts = &ctx->dyts;
    Tab *dyepeps = &ctx->dyepeps;
    Hash dyt_hash = ctx->dyt_hash;
    Hash dyepep_hash = ctx->dyepep_hash;
    Size n_flu_bytes = sizeof(DyeType) * n_aas;
    Size n_new_dyts = 0;
    Size n_new_dyepeps = 0;

    if(ctx->count_only) {
        // Add one record to both dyt and dyepeps
        if(dyts->n_rows == 0) {
            tab_add(dyts, 0, TAB_NO_LOCK);
        }
        if(dyepeps->n_rows == 0) {
            tab_add(dyepeps, 0, TAB_NO_LOCK);
        }
    }

    DyeType *flu = (DyeType *)alloca(n_flu_bytes);
    DyeType *working_flu = (DyeType *)alloca(n_flu_bytes);
    PIType *pi_bright = (PIType *)alloca(sizeof(PIType) * n_aas);
    for(Index aa_i = 0; aa_i < n_aas; aa_i++) {
        tab_var(PCB, pcb_row, pcb_block, aa_i);

        ensure_only_in_debug(
            (Index)pcb_row->pep_i == pep_i,
            "Mismatching pep_i in pcb_row pep_i=%ld row_pep_i=%ld aa_i=%ld",
            pep_i,
            (Index)pcb_row->pep_i,
            aa_i);

        Float64 f_ch_i = isnan(pcb_row->ch_i) ? (Float64)(N_MAX_CHANNELS - 1) : (pcb_row->ch_i);
        ensure_only_in_debug(0 <= f_ch_i && f_ch_i < N_MAX_CHANNELS, "f_ch_i out of bounds");
        flu[aa_i] = (DyeType)f_ch_i;

        Float64 p_bright = pcb_row->p_bright;
        p_bright = isnan(p_bright) ? 0.0 : p_bright;
        ensure_only_in_debug(
            0.0 <= p_bright && p_bright <= 1.0, "p_bright out of range pep_i=%ld aa_i=%ld %f", pep_i, aa_i, p_bright);
        pi_bright[aa_i] = prob_to_p_i(p_bright);

        working_flu[aa_i] = (DyeType)0;
    }

    // working_dyt is volatile stack copy of the out-going Dyt
    Size n_dyetrack_bytes = dyt_n_bytes(ctx->n_channels, ctx->n_cycles);
    Dyt *working_dyt = (Dyt *)alloca(n_dyetrack_bytes);
    memset(working_dyt, 0, n_dyetrack_bytes);

    Dyt *nul_dyt = (Dyt *)alloca(n_dyetrack_bytes);
    memset(nul_dyt, 0, n_dyetrack_bytes);

    // CHECK for unlabelled peptide
    int has_any_dye = 0;
    for(Index i = 0; i < n_aas; i++) {
        if(flu[i] != N_MAX_CHANNELS - 1) {
            has_any_dye = 1;
            break;
        }
    }

    Size n_dark_samples = 0;
    Size n_non_dark_samples = 0;
    while(has_any_dye && n_non_dark_samples < n_samples) {
        if(n_dark_samples > 10 * n_samples) {
            // Emergency exit. The recall is so low that we need to
            // just give up and declare that it can't be measured.
            n_dark_samples = 0;
            n_non_dark_samples = 0;
            break;
        }

        // GENERATE the working_dyetrack sample (Monte Carlo)
        //-------------------------------------------------------
        memcpy(working_flu, flu, n_flu_bytes);
        dyt_clear(working_dyt, n_channels, n_cycles);

        // MODEL dark-dyes (dyes dark before the first image)
        // These darks are the product of various dye factors which
        // are passed into this module already converted into PI form
        // (probability in 0 - max_unit64) by the pi_bright arrays
        for(Index aa_i = 0; aa_i < n_aas; aa_i++) {
            if(!rand64(pi_bright[aa_i])) {
                working_flu[aa_i] = NO_LABEL;
            }
        }

        Index head_i = 0;
        for(Index cy_i = 0; cy_i < n_cycles; cy_i++) {
            // EDMAN...
            // Edman degrdation chews off the N-terminal amino-acid.
            // With some peptide-attachment schemes, edman of the C-terminal AA isn't possible.
            // If successful this advances the "head_i" which is where we're summing from.
            if(cycles[cy_i] == CYCLE_TYPE_EDMAN) {
                if(rand64(pi_edman_success)) {
                    // always do rand64 to preserve RNG order independent of following condition
                    if(!prevent_edman_cterm || head_i < n_aas - 1) {
                        head_i++;
                    }
                }
            }

            // DETACH...
            // Detachment is when a peptide comes loose from the surface.
            // This means that all subsequent measurements go dark.
            if(rand64(pi_detach)) {
                for(Index aa_i = head_i; aa_i < n_aas; aa_i++) {
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
            for(Index aa_i = head_i; aa_i < n_aas; aa_i++) {
                ch_sums[working_flu[aa_i]]++;
            }
            for(Index ch_i = 0; ch_i < n_channels; ch_i++) {
                dyt_set_chcy(working_dyt, ch_sums[ch_i], n_channels, n_cycles, ch_i, cy_i);
            }

            // TODO: Explain why BLEACH is treated POST image
            //   Ie only makes a difference to the first cycle
            // BLEACH
            for(Index aa_i = head_i; aa_i < n_aas; aa_i++) {
                // For all REMAINING dyes (head_i:...) give
                // each dye a chance to photobleach.
                // TODO: Profile which is better, the branch here or just letting it over-write
                if(working_flu[aa_i] < NO_LABEL && rand64(pi_bleach)) {
                    working_flu[aa_i] = NO_LABEL;
                }
            }
        }

        // At this point we have the flu sampled into working_dyt
        // Now we look it up in the hash tabs.
        //-------------------------------------------------------

        if(memcmp(working_dyt, nul_dyt, n_dyetrack_bytes) == 0) {
            // The row was empty, note this and continue to try another sample
            n_dark_samples++;
            continue;
        }

        n_non_dark_samples++;

        HashKey dyt_hashkey = dyt_get_hashkey(working_dyt, n_channels, n_cycles);
        HashRec *dyt_hash_rec = hash_get(dyt_hash, dyt_hashkey);
        Dyt *dyt;
        ensure(dyt_hash_rec != (HashRec *)0, "dyt_hash full");
        if(dyt_hash_rec->key == 0) {
            // New record
            n_new_dyts++;
            Index dyt_i = 0;
            if(!ctx->count_only) {
                dyt_i = tab_add(dyts, working_dyt, ctx->n_threads > 1 ? ctx->tab_lock : TAB_NO_LOCK);
            }
            dyt = tab_ptr(Dyt, dyts, dyt_i);
            dyt_hash_rec->key = dyt_hashkey;
            dyt->count++;
            dyt->dyt_i = dyt_i;
            dyt_hash_rec->val = dyt;
        } else {
            // Existing record
            // Because this is a MonteCarlo sampling it really doesn't
            // matter if we occasionally mis-count due to thread
            // contention therefore there is no lock here.
            dyt = (Dyt *)(dyt_hash_rec->val);
            tab_validate_only_in_debug(dyts, dyt);
            dyt->count++;
        }
        tab_validate_only_in_debug(dyts, dyt);

        // SAVE the (dyt_i, pep_i) into dyepeps
        // (or inc count if it already exists)
        //-------------------------------------------------------
        HashKey dyepep_hashkey = dyepep_get_hashkey(dyt_hashkey, pep_i);
        HashRec *dyepep_hash_rec = hash_get(dyepep_hash, dyepep_hashkey);
        ensure(dyepep_hash_rec != (HashRec *)0, "dyepep_hash full");
        if(dyepep_hash_rec->key == 0) {
            // New record
            // If this were used multi-threaded, this would be a race condition
            n_new_dyepeps++;
            Index dyepep_i = 0;
            if(!ctx->count_only) {
                dyepep_i = tab_add(dyepeps, NULL, ctx->n_threads > 1 ? ctx->tab_lock : TAB_NO_LOCK);
            }
            tab_var(DyePepRec, dyepep, dyepeps, dyepep_i);
            dyepep_hash_rec->key = dyepep_hashkey;
            dyepep->dyt_i = dyt->dyt_i;
            dyepep->pep_i = pep_i;
            dyepep->n_reads++;
            dyepep_hash_rec->val = dyepep;
        } else {
            // Existing record
            // Same argument as above
            DyePepRec *dpr = (DyePepRec *)dyepep_hash_rec->val;
            tab_validate_only_in_debug(dyepeps, dpr);
            dpr->n_reads++;
        }
    }

    if(n_dark_samples + n_non_dark_samples > 0) {
        ctx->pep_recalls[pep_i] = (double)n_non_dark_samples / (double)(n_dark_samples + n_non_dark_samples);
    } else {
        ctx->pep_recalls[pep_i] = (double)0.0;
    }

    Counts counts;
    counts.n_new_dyts = n_new_dyts;
    counts.n_new_dyepeps = n_new_dyepeps;
    return counts;
}

Index context_work_orders_pop(SimV2Context *ctx) {
    // NOTE: This return +1! So that 0 can be reserved.
    if(ctx->n_threads > 1) {
        pthread_mutex_lock(ctx->work_order_lock);
    }

    Index i = ctx->next_pep_i;
    ctx->next_pep_i++;

    if(ctx->n_threads > 1) {
        pthread_mutex_unlock(ctx->work_order_lock);
    }

    if(i < ctx->n_peps) {
        return i + 1;
    }
    return 0;
}

typedef struct {
    Index thread_i;
    SimV2Context *ctx;
    Sint64 pep_i_status;
    pthread_t id;
    int stop_requested;
} ThreadContext;
#define THREAD_STATE_STARTED (-2)
#define THREAD_STATE_DONE (-1)

void *context_work_orders_worker(void *_tctx) {
    // The worker thread. Pops off which pep to work on next
    // continues until there are no more work orders.
    ThreadContext *tctx = (ThreadContext *)_tctx;
    SimV2Context *ctx = tctx->ctx;
    if(ctx->count_only) {
        ensure(ctx->n_threads == 1, "n_therads must be 1 when counting");
        trace("Counting n_dyts and n_dyepeps for %ld peps\n", ctx->n_peps);
        trace("pep_i, n_dyts, n_dyepeps\n");
    }
    Size n_dyts = 0;
    Size n_dyepeps = 0;
    while(!tctx->stop_requested) {
        Index pep_i_plus_1 = context_work_orders_pop(ctx);
        if(pep_i_plus_1 == 0) {
            break;
        }
        Index pep_i = pep_i_plus_1 - 1;
        Index pcb_i = tab_get(Index, &ctx->pep_i_to_pcb_i, pep_i);
        Index pcb_i_plus_1 = tab_get(Index, &ctx->pep_i_to_pcb_i, pep_i + 1);
        Size n_aas = pcb_i_plus_1 - pcb_i;
        Tab pcb_block = tab_subset(&ctx->pcbs, pcb_i, n_aas);

        Counts counts = context_sim_flu(ctx, pep_i, &pcb_block, n_aas);

        n_dyts += counts.n_new_dyts;
        n_dyepeps += counts.n_new_dyepeps;
        if(ctx->count_only && pep_i % 100 == 0) {
            trace("%ld, %ld, %ld\n", pep_i, n_dyts, n_dyepeps);
        }
        tctx->pep_i_status = (Sint64)pep_i;
    }
    if(ctx->count_only) {
        ctx->output_n_dyts = n_dyts;
        ctx->output_n_dyepeps = n_dyepeps;
    }
    tctx->pep_i_status = THREAD_STATE_DONE;
    return (void *)0;
}

// TODO: separate init and run - see sigproc_v2 for naming convention
int context_work_orders_start(SimV2Context *ctx) {
    // Allocate memory (this used to take place in the pyx file)
    uint8_t *dyts_buf = calloc(ctx->n_max_dyts, ctx->n_dyt_row_bytes);
    uint8_t *dyepeps_buf = calloc(ctx->n_max_dyepeps, sizeof(DyePepRec));
    HashRec *dyt_hash_buf = calloc(ctx->n_max_dyt_hash_recs, sizeof(HashRec));
    HashRec *dyepep_hash_buf = calloc(ctx->n_max_dyepep_hash_recs, sizeof(HashRec));

    ctx->work_order_lock = malloc(sizeof(pthread_mutex_t));
    ctx->tab_lock = malloc(sizeof(pthread_mutex_t));

    ctx->dyts = tab_by_n_rows(dyts_buf, ctx->n_max_dyts, ctx->n_dyt_row_bytes, TAB_GROWABLE);
    ctx->dyepeps = tab_by_size(dyepeps_buf, ctx->n_max_dyepeps * sizeof(DyePepRec), sizeof(DyePepRec), TAB_GROWABLE);
    ctx->dyt_hash = hash_init(dyt_hash_buf, ctx->n_max_dyt_hash_recs);
    ctx->dyepep_hash = hash_init(dyepep_hash_buf, ctx->n_max_dyepep_hash_recs);
    ctx->pep_i_to_pcb_i = tab_by_n_rows(ctx->pep_i_to_pcb_i_buf, ctx->n_peps + 1, sizeof(Index), TAB_NOT_GROWABLE);

    // context_dump(ctx);

    // Initialize mutex and start the worker thread(s).
    ensure(setup_and_sanity_check(ctx->n_channels, ctx->n_cycles) == 0, "Sanity checks failed");
    rand64_seed(ctx->rng_seed);

    ctx->next_pep_i = 0;

    // Add a nul-row
    Size n_dyetrack_bytes = dyt_n_bytes(ctx->n_channels, ctx->n_cycles);
    Dyt *nul_rec = (Dyt *)alloca(n_dyetrack_bytes);
    ensure(nul_rec != NULL, "alloca failed");
    memset(nul_rec, 0, n_dyetrack_bytes);
    HashKey dyt_hashkey = dyt_get_hashkey(nul_rec, ctx->n_channels, ctx->n_cycles);
    HashRec *dyt_hash_rec = hash_get(ctx->dyt_hash, dyt_hashkey);
    ensure(dyt_hash_rec->key == 0, "dyt hash should not have found nul row");

    Tab *dyts = &ctx->dyts;
    Index nul_i = tab_add(dyts, nul_rec, TAB_NO_LOCK);
    tab_var(Dyt, nul_dyt, dyts, nul_i);
    dyt_hash_rec->key = dyt_hashkey;
    nul_dyt->count++;
    nul_dyt->dyt_i = nul_i;
    dyt_hash_rec->val = nul_dyt;

    // TODO: remove threading from c - see radiometry.py:200

    ThreadContext thread_contexts[256];
    ensure(0 < ctx->n_threads && ctx->n_threads < 256, "Invalid n_threads");

    if(ctx->n_threads > 1) {
        int ret = pthread_mutex_init(ctx->work_order_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");

        ret = pthread_mutex_init(ctx->tab_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");
    }

    for(Index thread_i = 0; thread_i < ctx->n_threads; thread_i++) {
        thread_contexts[thread_i].thread_i = thread_i;
        thread_contexts[thread_i].ctx = ctx;
        thread_contexts[thread_i].pep_i_status = THREAD_STATE_STARTED;
        thread_contexts[thread_i].stop_requested = 0;
        int ret =
            pthread_create(&thread_contexts[thread_i].id, NULL, context_work_orders_worker, &thread_contexts[thread_i]);
        ensure(ret == 0, "Thread not created.");
    }

    // MONITOR progress and callback from this main thread
    // Python doesn't seem to like callbacks coming from other threads
    int interrupted = 0;
    while(1) {
        Size n_threads_done = 0;
        Sint64 largest_pep_i_done = 0;
        for(Index thread_i = 0; thread_i < ctx->n_threads; thread_i++) {
            if(thread_contexts[thread_i].pep_i_status == THREAD_STATE_DONE) {
                n_threads_done++;
            }
            largest_pep_i_done = max(largest_pep_i_done, thread_contexts[thread_i].pep_i_status);
        }
        if(n_threads_done == ctx->n_threads) {
            break;
        }
        if(largest_pep_i_done > 0 && largest_pep_i_done % 100 == 0) {
            ctx->progress_fn(largest_pep_i_done, ctx->n_peps, 0);
        }
        int got_interrupt = ctx->check_keyboard_interrupt_fn();
        if(got_interrupt) {
            for(Index thread_i = 0; thread_i < ctx->n_threads; thread_i++) {
                thread_contexts[thread_i].stop_requested = 1;
            }
            interrupted = 1;
            break;
        }
        usleep(10000); // 10 ms
    }

    for(Index thread_i = 0; thread_i < ctx->n_threads; thread_i++) {
        pthread_join(thread_contexts[thread_i].id, NULL);
    }

    return 0;
}

int context_free(SimV2Context *ctx) {
    free(ctx->work_order_lock);
    free(ctx->tab_lock);
    free(ctx->dyts.base);
    free(ctx->dyepeps.base);
    free(ctx->dyt_hash.recs);
    free(ctx->dyepep_hash.recs);
}

Size context_dyt_get_count(SimV2Context *ctx, Index dyt_i) {
    tab_var(Dyt, dyt, &ctx->dyts, dyt_i);
    return dyt->count;
}

DyeType *context_dyt_dyetrack(SimV2Context *ctx, Index dyt_i) {
    tab_var(Dyt, dyt, &ctx->dyts, dyt_i);
    return dyt->chcy_dye_counts;
}

DyePepRec *context_dyepep(SimV2Context *ctx, Index dyepep_i) { return tab_ptr(DyePepRec, &ctx->dyepeps, dyepep_i); }

void context_dump(SimV2Context *ctx) {
    printf("SimV2Context:\n");
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
