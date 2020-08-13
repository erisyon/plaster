#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "math.h"
#include "c_survey_v2_fast.h"


/*

Looks like there's more and more common c code
Need to get a common pxd file
Change all dye_ and dt stuff to consistent dyt_
I really want more than one version of table_init

mlpep = maximum likely peptide

Build a "dyt_to_mlpep" LUT

for each peptide (parallel)
    extract out all of the dyts for that peptide (a groupby in pandas or a index similar to sim_v2)
        Lookup n_neighbors for each of those dyetracks
        that's a matrix of n_dyt, n_neighbor
        there's a parallel dist vector returned from flann
        replace each value in that table with a LUT from dyt_to_mlpep
        Search along the columns for the first pep that isn't THIS pep
        That pep is the closest interference pep

        Multiply f(dist) * frac_of_reads_to_this_dyt
        Add all those up and that's the isolation metric. A big number is better.
*/

void context_pep_measure_isolation(
    Context *ctx,
    Index pep_i,
) {
    /*
    TODO.
    A large value of isolation means the peptide is well separated.
    */

    int n_neighbors = ctx->n_neighbors;
    int dyt_row_n_bytes = ctx->dyemat->n_bytes_per_row;

    // SETUP a local table for the dyepeps of this peptide.
    Index dyepeps_offset_start_of_this_pep = *table_get_row(&ctx->pep_i_to_dyepep_row_i, pep_i);
    Index dyepeps_offset_start_of_next_pep = *table_get_row(&ctx->pep_i_to_dyepep_row_i, pep_i + 1);
    Size n_dyts = dyepeps_offset_start_of_next_pep - dyepeps_offset_start_of_this_pep;
    Table dyepeps = table_init_subset(&ctx->dyepeps, dyepeps_offset_start, n_dyts, 1);

    // ALLOC a dyemat for all of the dyts of this peptide
    DyeType *local_dyemat_buffer = (DyeType *)alloca(dyt_row_n_bytes * n_dyts);
    Table local_dyemat = table_init(local_dyemat_buffer, dyt_row_n_bytes * n_dyts, dyt_row_n_bytes);

    // LOAD the local dyemat table by copying using the dyt_iz in dyepeps
    for(Index i=0; i<n_dyts; i++) {
        DyePepRec *dyepep_row = table_get_row(&dyepeps, i, DyePepRec);

        DyeType *src = table_get_row(&ctx->dyemat, dyepep_row->dtr_i, DyeType);
        DyeType *dst = table_get_row(local_dyemat, i, DyeType);
        memcpy(dst, src, dyt_row_n_bytes);
    }

    // ALLOC space for the NN table
    Size nn_dyt_iz_row_n_bytes = n_neighbors * sizeof(int);
    Size nn_dists_row_n_bytes = n_neighbors * sizeof(float);
    int *nn_dyt_iz_buf = (int *)alloca(n_dyts * nn_dyt_iz_row_n_bytes);
    float *nn_dists_buf = (float *)alloca(n_dyts * nn_dists_row_n_bytes);
    memset(nn_dyt_iz_buf, 0, n_dyts * nn_dyt_iz_row_n_bytes);
    memset(nn_dists_buf, 0, n_dyts * nn_dists_row_n_bytes);

    Table nn_dyt_iz = table_init(nn_dyt_iz_buf, n_dyts * nn_dyt_iz_row_n_bytes, nn_dyt_iz_row_n_bytes);
    Table nn_dists = table_init(nn_dists_buf, n_dyts * nn_dists_row_n_bytes, nn_dists_row_n_bytes);

    // FETCH a batch of neighbors from FLANN in one call.
    flann_find_nearest_neighbors_index_float(
        ctx->flann_index_id,
        table_get_row(local_dyemat, 0, DyeType),
        n_dyts,
        nn_dyt_iz_buf,
        nn_dists_buf,
        n_neighbors,
        &ctx->flann_params
    );

    /*
    replace each value in that table with a LUT from dyt_to_mlpep
    Search along the columns for the first pep that isn't THIS pep
    That pep is the closest interference pep

    Multiply f(dist) * frac_of_reads_to_this_dyt
    Add all those up and that's the isolation metric. A big number is better.
    */

    // FOLLOW each neighbor dyt to its mlpep. Often, this mlpep will be the
    // same as the pep_i that is currently analyzeing. The first mlpep that
    // is NOT this smae peptide we call the most-in-contention peptide (mic_pep_i).
    // The distance to that micpep is a distance of interest.

    Size n_reads_total = 0;
    IsolationType isolation_metric = (ContentionType)0;
    for (Index i=0; i<n_dyts; i++) {
        DyePepRec *dyepep_row = table_get_row(&dyepeps, i, DyePepRec);

        Index mic_pep_i = 0;
        Float32 mic_pep_dist = (Float32)0;
        Index *nn_dyt_row_i = table_get_row(&nn_dyt_iz, i, Index);
        float *nn_dists_row_i = table_get_row(&nn_dists, i, float);
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Index dyt_i = nn_dyt_row_i[nn_i];
            ensure_only_in_debug(0 <= dyt_i && dyt_i < ctx->dyemat.n_rows, "Illegal dyt in nn lookup");

            // LOOKUP the mlpep for this dyt
            Index mlpep_i = *table_get_row(&ctx->dyt_i_to_mlpep_i, dyt_i, Index);
            ensure_only_in_debug(0 <= mlpep_i && mlpep_i < ctx->n_peps);

            if(mlpep_i != pep_i) {
                // Found the first neighbor dyetrack with a Max-Liklihood peptide that isn't the current peptide
                mic_pep_i = mlpep_i;
                mic_pep_dist = nn_dists_row_i[nn_i];
                break
            }
        }
        if(nn_i == n_neighbors) {
            // Got to the end without finding another peptide in the neighbor list
            // We say that this peptide has no contention but this is a problem because
            // we sum the contributions of each distance to the mlpep and a
            // LARGER value is a better isolation. But here we know that the closest
            // one is really far but we don't know HOW far so we don't know how
            // to scale it.
            // Thus, this value has to get passed in from context and the value
            // probably has to be determined by sampling.
            mic_pep_dist = (Float32)(
                ctx->distance_to_assign_an_isolated_pep
                * ctx->distance_to_assign_an_isolated_pep
            );
        }

        isolation_metric += dyepep_row->n_reads * sqrt(mic_pep_dist);
        n_reads_total += dyepep_row->n_reads;
    }

    // RECORD the result
    IsolationType *result = table_get_row(&ctx->output_pep_i_to_isolation_metric, pep_i, IsolationType);
    *result = isolation_metric / (IsolationType)n_reads_total;
}


Index context_work_orders_pop(Context *ctx) {
    // TODO: This could be dried with similar sim_v2 code
    // (but remember they refer to differnte Context structs)
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
    Size n_dtrs = 0;
    Size n_dyepeps = 0;
    while(1) {
        Index pep_i_plus_1 = context_work_orders_pop(ctx);
        if(pep_i_plus_1 == 0) {
            break;
        }
        Index pep_i = pep_i_plus_1 - 1;

        context_measure_peptide_isolation(ctx, pep_i);

        if(pep_i % 100 == 0) {
            ctx->progress_fn(pep_i, ctx->n_peps, 0);
        }
    }
    ctx->progress_fn(ctx->n_peps, ctx->n_peps, 0);
    ctx->output_n_dtrs = n_dtrs;
    ctx->output_n_dyepeps = n_dyepeps;
    return (void *)0;
}


void context_start(Context *ctx) {
    // Initialize mutex and start the worker thread(s).
    ctx->next_pep_i = 0;

    // TODO: DRY with simialr code in nn_v2

    ensure(
        ctx->n_neighbors <= ctx->dyemat.n_rows,
        "FLANN does not support requesting more neihbors than there are data points"
    );
    ensure(
        ctx->n_neighbors <= N_MAX_NEIGHBORS,
        "Too many neighbors requested"
    );

    // CLEAR internally controlled elements
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;
    ctx->flann_index_id = 0;

    // CREATE the ANN index
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_float(
        table_get_row(&ctx->dyemat, 0, DyeType),
        ctx->dyemat.n_rows,
        ctx->n_dyt_cols,
        &speedup,
        &ctx->flann_params
    );

    // START threads
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

}

