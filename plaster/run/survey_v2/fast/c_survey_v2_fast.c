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

void dump_row_of_dyemat(Table dyemat, int row) {
    DyeType *dyt = table_get_row(&dyemat, row, DyeType);
    for (Index k=0; k<dyemat.n_bytes_per_row; k++) {
        printf("%d ", dyt[k]);
    }
    printf("\n");
}

void context_pep_measure_isolation(SurveyV2FastContext *ctx, Index pep_i) {
    /*
    TODO.
    A large value of isolation means the peptide is well separated.
    */

    Size n_global_dyts = ctx->n_dyts;
    int n_neighbors = ctx->n_neighbors;
    int dyt_row_n_bytes = ctx->dyemat.n_bytes_per_row;
    int n_dyt_cols = ctx->n_dyt_cols;
    ensure(n_dyt_cols > 0, "no n_dyt_cols");

    // SETUP a local table for the dyepeps of this peptide.
    Index dyepeps_offset_start_of_this_pep = *table_get_row(&ctx->pep_i_to_dyepep_row_i, pep_i, Index);
    Index dyepeps_offset_start_of_next_pep = *table_get_row(&ctx->pep_i_to_dyepep_row_i, pep_i + 1, Index);
    Size n_local_dyts = dyepeps_offset_start_of_next_pep - dyepeps_offset_start_of_this_pep;
    ensure(n_local_dyts > 0, "no dyts pep_i=%ld (%ld %ld)", pep_i, dyepeps_offset_start_of_next_pep, dyepeps_offset_start_of_this_pep);
    Table dyepeps = table_init_subset(&ctx->dyepeps, dyepeps_offset_start_of_this_pep, n_local_dyts, 1);

    // TODO: This table_init_readonly in the following contexts is a misnomer,
    //   I need to rename it. What I mean is that the table isn't going to GROW
    // ALLOC a dyemat for all of the dyts of this peptide
    RadType *local_dyemat_buffer = (RadType *)alloca(n_local_dyts * n_dyt_cols * sizeof(DyeType));
    memset(local_dyemat_buffer, 0, n_local_dyts * n_dyt_cols * sizeof(DyeType));
    Table local_dyemat = table_init_readonly(local_dyemat_buffer, n_local_dyts * n_dyt_cols * sizeof(DyeType), n_dyt_cols * sizeof(DyeType));

    // LOAD the local dyemat table by copying using the dyt_iz in dyepeps
    for(Index i=0; i<n_local_dyts; i++) {
        DyePepRec *dyepep_row = table_get_row(&dyepeps, i, DyePepRec);

        DyeType *src = table_get_row(&ctx->dyemat, dyepep_row->dtr_i, DyeType);
        DyeType *dst = table_get_row(&local_dyemat, i, DyeType);
        memcpy(dst, src, dyt_row_n_bytes);
    }

    // ALLOC space for the NN table
    Size nn_dyt_iz_row_n_bytes = n_neighbors * sizeof(int);
    Size nn_dists_row_n_bytes = n_neighbors * sizeof(float);
    int *nn_dyt_iz_buf = (int *)alloca(n_local_dyts * nn_dyt_iz_row_n_bytes);
    float *nn_dists_buf = (float *)alloca(n_local_dyts * nn_dists_row_n_bytes);
    memset(nn_dyt_iz_buf, 0, n_local_dyts * nn_dyt_iz_row_n_bytes);
    memset(nn_dists_buf, 0, n_local_dyts * nn_dists_row_n_bytes);

    // Remember: nn_dyt_iz_buf contains offets into the GLOBAL dyemat
    Table nn_dyt_iz = table_init_readonly(nn_dyt_iz_buf, n_local_dyts * nn_dyt_iz_row_n_bytes, nn_dyt_iz_row_n_bytes);
    Table nn_dists = table_init_readonly(nn_dists_buf, n_local_dyts * nn_dists_row_n_bytes, nn_dists_row_n_bytes);

    // FETCH a batch of neighbors from FLANN in one call against the GLOBAL index of dyetracks
    int ret = flann_find_nearest_neighbors_index_byte(
        ctx->flann_index_id,
        table_get_row(&local_dyemat, 0, DyeType),
        n_local_dyts,
        nn_dyt_iz_buf,
        nn_dists_buf,
        n_neighbors,
        &ctx->flann_params
    );
    ensure(ret == 0, "flann returned error code");

    // Sanity check
    for (Index i=0; i<n_local_dyts; i++) {
        int *row = table_get_row(&nn_dyt_iz, i, int);
        for (int j=0; j<n_neighbors; j++ ) {
            //printf("%3d ", row[j]);
            if(!(0 <= row[j] && row[j] < (int)n_global_dyts)) {
                trace("\nfield in row %d %d %d %d\n", i, j, row[j], n_global_dyts);
            }
        }
        //printf("\n");
    }

    // FOLLOW each neighbor dyt to its mlpep. Often, this mlpep will be the
    // same as the pep_i that is currently analyzeing. The first mlpep that
    // is NOT this same peptide we call the most-in-contention peptide (mic_pep_i).
    // The distance to that micpep is a distance of interest.

    Size n_neighbors_u = (Size)n_neighbors;
    Size n_reads_total = 0;
    IsolationType isolation_metric = (IsolationType)0;
    for (Index i=0; i<n_local_dyts; i++) {
        DyePepRec *dyepep_row = table_get_row(&dyepeps, i, DyePepRec);

        Float32 mic_pep_dist = (Float32)0;
        int *nn_dyt_row_i = table_get_row(&nn_dyt_iz, i, int);
        float *nn_dists_row_i = table_get_row(&nn_dists, i, float);

        Index nn_i = 0;
        for (nn_i=0; nn_i<n_neighbors_u; nn_i++) {
            int global_dyt_i = nn_dyt_row_i[nn_i];
            ensure_only_in_debug(0 <= global_dyt_i && global_dyt_i < (int)n_global_dyts, "Illegal dyt in nn lookup: %ld %ld", global_dyt_i, n_global_dyts);

            // LOOKUP the mlpep for this dyt_i. remember, we must CONVERT from local_dyt_i to global_dyt_i
            Index mlpep_i = *table_get_row(&ctx->dyt_i_to_mlpep_i, global_dyt_i, Index);
            ensure_only_in_debug(0 <= mlpep_i && mlpep_i < ctx->n_peps, "mlpep_i out of bounds %ld %ld", mlpep_i, ctx->n_peps);

            if(mlpep_i != pep_i) {
                // Found the first neighbor dyetrack with a Max-Liklihood peptide that isn't the current peptide
                mic_pep_dist = nn_dists_row_i[nn_i];
                break;
            }
        }

        if(nn_i == n_neighbors_u) {
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


Index context_work_orders_pop(SurveyV2FastContext *ctx) {
    // TODO: This could be dried with similar sim_v2 code
    // (but remember they refer to differnte SurveyV2FastContext structs)
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
    SurveyV2FastContext *ctx = (SurveyV2FastContext *)_ctx;
    while(1) {
        Index pep_i_plus_1 = context_work_orders_pop(ctx);
        if(pep_i_plus_1 == 0) {
            break;
        }
        Index pep_i = pep_i_plus_1 - 1;

        context_pep_measure_isolation(ctx, pep_i);

        if(pep_i % 100 == 0) {
            ctx->progress_fn(pep_i, ctx->n_peps, 0);
        }
    }
    ctx->progress_fn(ctx->n_peps, ctx->n_peps, 0);
    return (void *)0;
}


void context_start(SurveyV2FastContext *ctx) {
    // Initialize mutex and start the worker thread(s).
    ctx->next_pep_i = 0;

    // TODO: DRY with simialr code in nn_v2

    ensure(
        ctx->n_neighbors <= ctx->dyemat.n_rows,
        "FLANN does not support requesting more neihbors than there are data points"
    );

    // CLEAR internally controlled elements
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;
    ctx->flann_index_id = 0;

    // CREATE the ANN index
    // TODO: DRY with NN
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_byte(
        table_get_row(&ctx->dyemat, 0, DyeType),
        ctx->dyemat.n_rows,
        ctx->n_dyt_cols,
        &speedup,
        &ctx->flann_params
    );

// HACK TEST
//#define n_neighbors (2)
//int nn_dyt_iz_buf[n_neighbors];
//float nn_dists_buf[n_neighbors];
//int ret = flann_find_nearest_neighbors_index_byte(
//    ctx->flann_index_id,
//    table_get_row(&ctx->dyemat, 3, DyeType),
//    1,
//    nn_dyt_iz_buf,
//    nn_dists_buf,
//    n_neighbors,
//    &ctx->flann_params
//);
//ensure(ret == 0, "flann returned error code");
//for (int j=0; j<n_neighbors; j++) {
//    Index dyt_i = nn_dyt_iz_buf[j];
//    printf("dyt_i=%ld\n", dyt_i);
//    DyeType *dyt = table_get_row(&ctx->dyemat, dyt_i, DyeType);
//    for (Index k=0; k<ctx->n_dyt_cols; k++) {
//        printf("%d ", dyt[k]);
//    }
//    printf("\n");
//}
//printf("\n");


    // START threads
    pthread_t ids[256];
    ensure(0 < ctx->n_threads && ctx->n_threads < 256, "Invalid n_threads");

    if(ctx->n_threads > 1) {
        int ret = pthread_mutex_init(&ctx->work_order_lock, NULL);
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
