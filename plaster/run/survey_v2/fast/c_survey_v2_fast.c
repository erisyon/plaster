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

Survey

This code's job is to measure how "isolated" each peptide is
with the goal of quickly prediciting how well a given
label/protease scheme will perform.

*/

void dump_dyepeps(SurveyV2FastContext *ctx) {
    Index last_pep_i = 0xFFFFFFFFFFFFFFFF;
    for(Index i=0; i<ctx->dyepeps.n_rows; i++) {
        tab_var(DyePepRec, dyepep, &ctx->dyepeps, i);
        tab_var(Index, mlpep_i, &ctx->dyt_i_to_mlpep_i, dyepep->dtr_i);

        if(last_pep_i != dyepep->pep_i) {
            trace("pep_i%lu\n", dyepep->pep_i);
            last_pep_i = dyepep->pep_i;
        }

        //if(*mlpep_i != dyepep->pep_i) {
            trace("  dyt_i:%-4lu n_reads:%-8lu  mlpep_i:%-4lu   ", dyepep->dtr_i, dyepep->n_reads, *mlpep_i);

            tab_var(DyeType, dyt, &ctx->dyemat, dyepep->dtr_i);
            for (Index k=0; k<ctx->dyemat.n_bytes_per_row; k++) {
                trace("%d ", dyt[k]);
            }
            trace("\n");
        //}
    }
}

void dump_row_of_dyemat(Tab *dyemat, int row, char *prefix) {
    tab_var(DyeType, dyt, dyemat, row);
    for (Index k=0; k<dyemat->n_bytes_per_row; k++) {
        printf("%s%d ", prefix, dyt[k]);
    }
    printf("\n");
}

void context_pep_measure_isolation(SurveyV2FastContext *ctx, Index pep_i) {
    /*

    Terminology:
        dyt: Dyetrack
        ml: Most Likely
        mic: Most In Contention
        pep: Peptide
        nn: nearest neighbor
        local_dyts: The dyetracks that are associated with pep_i
            (ie "local" because they pertain only to the input parameter)
        global:dyts: The set of all dyetracks in the simulation.
        ml_pep: "Most Likely Peptide" That is the peptide with the most reads
            from any given dyetrack
        self-dyetrack: a dyetrack that has THIS peptide as its ML-pep
        foreign-dyetrack: a dyetrack that has SOME OTHER peptide as its ML-pep
        isolation: A metric of how well separated this peptide is
            from other peptides. This is a relative metric, not
            an actual distance (ie this is NOT a Euclidiean distance)
        contention: The inverse of isolation. A large number means the
            peptide is LESS isolated.

    This funciton analyzes the "isolation" of a single peptide.
    It has access to:
        * The dyepeps which is a table w/ columns: (dyt_i, pep_i, n_reads)
            That is, each peptide has a list of all dyetracks it can
            create and how many reads that peptide generated for each
            dyetrack.
        * All the dyetracks

    We seek features for this peptide:
         * A measure of "isolation" (bigger number means better isolated)
         * Which OTHER peptide is the most contentious with this peptide?

    Algorithm:
        For this peptide, consider all dyetracks and measure their distance
        to their closest neighbor dyetrack.
        Scale those dyetrack distances by the n_reads for that dyetrack
        Sum all those read-scaled distances up and call that the
        "isolation metric"

        Meanwhile, compute the contention metric for the ml-peptide
        for each dyetrack.
        Sum those ml-pep contentions over every dyetrack
        Find the "most contentious" peptide to return

        In python-like code:

        isolation_by_dyt_i = {}
        contention_by_pep_i = {}
        for (
            dyt_i,
            distance_to_closest_dyt_with_a_foreign_mlpep,
            n_reads_to_dyt_i,
            closest_foreign_mlpep_i
        ) in this_peptide_dyts:
            isolation = n_reads_to_dyt_i * distance_to_closest_dyt_with_a_foreign_mlpep
            contention = n_reads_to_dyt_i / distance_to_closest_dyt_with_a_foreign_mlpep

            isolation_by_dyt_i[dyt_i] += isolation
            contention_by_pep_i[closest_foreign_mlpep_i] += contention

        total_isolation_for_this_pep = sum( isolation_of_dyt_i )
        most_in_contention_pep = the_peptide_with_the_highest_contention_sum(contention_by_pep_i)
    */

    Size n_global_dyts = ctx->n_dyts;
    int n_neighbors = ctx->n_neighbors;
    int dyt_row_n_bytes = ctx->dyemat.n_bytes_per_row;
    int n_dyt_cols = ctx->n_dyt_cols;
    ensure(n_dyt_cols > 0, "no n_dyt_cols");

    // SETUP a local table for the dyepeps OF THIS peptide by using the
    // pep_i_to_dyepep_row_i table to get the start and stop range.
    tab_var(Index, dyepeps_offset_start_of_this_pep, &ctx->pep_i_to_dyepep_row_i, pep_i);
    tab_var(Index, dyepeps_offset_start_of_next_pep, &ctx->pep_i_to_dyepep_row_i, pep_i + 1);
    int _n_local_dyts = *dyepeps_offset_start_of_next_pep - *dyepeps_offset_start_of_this_pep;
    ensure(_n_local_dyts > 0, "no dyts pep_i=%ld (this=%ld next=%ld)", pep_i, *dyepeps_offset_start_of_this_pep, *dyepeps_offset_start_of_next_pep);
    Index n_local_dyts = (Index)_n_local_dyts;

    // Using the pep_i_to_dyepep_row_i we now have the range of the dyepeps and we
    // can create a table subset (which is jsut a view into the table)
    Tab dyepeps = tab_subset(&ctx->dyepeps, *dyepeps_offset_start_of_this_pep, n_local_dyts);

    // We need a contiguous dyemat to feed to the FLANN function so we have
    // to copy each referenced dyemat from the global ctx->dyemat into a local copy.
    // ALLOC a dyemat for all of the dyts of this peptide
    RadType *local_dyemat_buffer = (RadType *)alloca(n_local_dyts * n_dyt_cols * sizeof(DyeType));
    memset(local_dyemat_buffer, 0, n_local_dyts * n_dyt_cols * sizeof(DyeType));
    Tab local_dyemat = tab_by_n_rows(local_dyemat_buffer, n_local_dyts, n_dyt_cols * sizeof(DyeType), TAB_NOT_GROWABLE);

    // LOAD the local dyemat table by copying using the dyt_iz in dyepeps
    for(Index i=0; i<n_local_dyts; i++) {
        tab_var(DyePepRec, dyepep_row, &dyepeps, i);

        tab_var(DyeType, src, &ctx->dyemat, dyepep_row->dtr_i);
        tab_var(DyeType, dst, &local_dyemat, i);
        memcpy(dst, src, dyt_row_n_bytes);
    }

    // FLANN needs buffers to write what it found as the closest neighbors and their distances.
    // ALLOC space for those table on the stack because they shouldn't be too large.
    Size nn_dyt_iz_row_n_bytes = n_neighbors * sizeof(int);
    Size nn_dists_row_n_bytes = n_neighbors * sizeof(float);
    int *nn_dyt_iz_buf = (int *)alloca(n_local_dyts * nn_dyt_iz_row_n_bytes);
    float *nn_dists_buf = (float *)alloca(n_local_dyts * nn_dists_row_n_bytes);
    memset(nn_dyt_iz_buf, 0, n_local_dyts * nn_dyt_iz_row_n_bytes);
    memset(nn_dists_buf, 0, n_local_dyts * nn_dists_row_n_bytes);

    Tab nn_dyt_iz = tab_by_n_rows(nn_dyt_iz_buf, n_local_dyts, nn_dyt_iz_row_n_bytes, TAB_NOT_GROWABLE);
    Tab nn_dists = tab_by_n_rows(nn_dists_buf, n_local_dyts, nn_dists_row_n_bytes, TAB_NOT_GROWABLE);

    // FETCH a batch of neighbors from FLANN in one call against the GLOBAL index of dyetracks
    int ret = flann_find_nearest_neighbors_index_byte(
        ctx->flann_index_id,
        tab_ptr(DyeType, &local_dyemat, 0),
        n_local_dyts,
        nn_dyt_iz_buf,
        nn_dists_buf,
        n_neighbors,
        &ctx->flann_params
    );
    ensure(ret == 0, "flann returned error code");

    // At this point FLANN has found neighbors (and their distances) for each local dyetrack
    // Tab nn_dyt_iz contains the GLBOAL dyt_i index for each neighbor
    // Tab nn_dists contains the distance
    // TODO: Check FLANN retuns the square of the distance (?)

    // Sanity check
//    for (Index i=0; i<n_local_dyts; i++) {
//        tab_var(int, row, &nn_dyt_iz, i);
//        for (int j=0; j<n_neighbors; j++ ) {
//            //printf("%3d ", row[j]);
//            if(!(0 <= row[j] && row[j] < (int)n_global_dyts)) {
//                trace("\nfield in row %d %d %d %d\n", i, j, row[j], n_global_dyts);
//            }
//        }
//        //printf("\n");
//    }

    // FOLLOW each neighbor dyt to its ml-pep. Often, this ml-pep will be the
    // same as the pep_i that we are currently analyzing. The first mlpep that
    // is NOT this same peptide we call the most-in-contention peptide (mic_pep_i).
    // The distance to that micpep is a distance of interest.

    Size n_neighbors_u = (Size)n_neighbors;
    Size n_reads_total = 0;
    Index mic_pep_i = 0;
//    Index mic_dyt_i = 0;

    Index *local_dyt_i_to_closest_mlpep_i_buf = (Index *)alloca(n_local_dyts * sizeof(Index));
    memset(local_dyt_i_to_closest_mlpep_i_buf, 0, n_local_dyts * sizeof(Index));
    Tab local_dyt_i_to_closest_mlpep_i = tab_by_n_rows(local_dyt_i_to_closest_mlpep_i_buf, n_local_dyts, sizeof(Index), TAB_NOT_GROWABLE);

    float *local_dyt_i_to_closest_dyt_isolation_buf = (float *)alloca(n_local_dyts * sizeof(float));
    memset(local_dyt_i_to_closest_dyt_isolation_buf, 0, n_local_dyts * sizeof(float));
    Tab local_dyt_i_to_closest_dyt_isolation = tab_by_n_rows(local_dyt_i_to_closest_dyt_isolation_buf, n_local_dyts, sizeof(float), TAB_NOT_GROWABLE);

    // For each dyt that pep_i could create, populate:
    //   local_dyt_i_to_closest_mlpep_i
    //   local_dyt_i_to_closest_dyt_isolation

    for (Index i=0; i<n_local_dyts; i++) {

        tab_var(DyePepRec, dyepep_row, &dyepeps, i);
//trace("  i=%lu  (dyt:%lu pep:%lu cnt:%lu) \n", i, dyepep_row->dtr_i, dyepep_row->pep_i, dyepep_row->n_reads);

        Float32 mic_pep_isolation = (Float32)0;
        tab_var(int, nn_dyt_row_i, &nn_dyt_iz, i);
        tab_var(float, nn_dists_row_i, &nn_dists, i);

       // For each Neighbor that this dyt i has we want to accumulate a
       // distance measurement -- ONLY IF that neighbor dyetrack mlpep
       // is some peptide OTHER THAN the current peptide.
       // (That is, dyetracks that come from the same peptide are not in contention)
       //
       // Hence we're searching for the closest neighbor that has a different mlpep than pep_i
       // Note that FLANN returns neighbors in closest first so we can break
       // out of the search loop as soon as we find a mlpep != pep

        Index nn_i = 0;
        for (nn_i=0; nn_i<n_neighbors_u; nn_i++) {
            int global_dyt_i = nn_dyt_row_i[nn_i];
            ensure_only_in_debug(0 <= global_dyt_i && global_dyt_i < (int)n_global_dyts, "Illegal dyt in nn lookup: %ld %ld", global_dyt_i, n_global_dyts);

            if((int)dyepep_row->dtr_i == global_dyt_i) {
                // Do not compare a dyetrack to itself, it will always be zero away
                continue;
            }

//trace("    nn_i=%lu (global_dyt_i=%lu)\n", nn_i, global_dyt_i);

            // LOOKUP the mlpep for this dyt_i. remember, we must CONVERT from local_dyt_i to global_dyt_i
            Index mlpep_i = tab_get(Index, &ctx->dyt_i_to_mlpep_i, global_dyt_i);
            ensure_only_in_debug(0 <= mlpep_i && mlpep_i < ctx->n_peps, "mlpep_i out of bounds %ld %ld", mlpep_i, ctx->n_peps);

            if(mlpep_i != pep_i) {
                // Found the first neighbor dyetrack with a Max-Liklihood peptide that isn't the current peptide
                mic_pep_isolation = nn_dists_row_i[nn_i];
                mic_pep_i = mlpep_i;
//                mic_dyt_i = global_dyt_i;
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
            mic_pep_isolation = (Float32)(
                ctx->distance_to_assign_an_isolated_pep
                * ctx->distance_to_assign_an_isolated_pep
            );
            mic_pep_i = 0;
//            mic_dyt_i = 0;
        }

//trace("      reads=%lu mic_pep_isolation=%f scaled=%f mic_pep_i=%lu  mic_dyt_i=%lu \n", dyepep_row->n_reads, mic_pep_isolation, dyepep_row->n_reads * sqrt(mic_pep_isolation), mic_pep_i, mic_dyt_i);
        mic_pep_isolation = dyepep_row->n_reads * sqrt(mic_pep_isolation);
        tab_set(&local_dyt_i_to_closest_dyt_isolation, i, &mic_pep_isolation);
        tab_set(&local_dyt_i_to_closest_mlpep_i, i, &mic_pep_i);

        n_reads_total += dyepep_row->n_reads;
    }

    // Now find the most in contention -- the peptide with the lowest isolation
    float smallest_isolation = 1e10f;
    Index closest_mlpep_i = 0;
    for (Index i=0; i<n_local_dyts; i++) {
        float isolation = tab_get(float, &local_dyt_i_to_closest_dyt_isolation, i);
        if(isolation < smallest_isolation) {
            smallest_isolation = isolation;
            closest_mlpep_i = tab_get(Index, &local_dyt_i_to_closest_mlpep_i, i);
        }
    }
//trace("  smallest_isolation=%f closest_mlpep_i=%lu  \n", smallest_isolation, closest_mlpep_i);

    // RECORD the result
    IsolationType result = smallest_isolation / (IsolationType)n_reads_total;
    tab_set(&ctx->output_pep_i_to_isolation_metric, pep_i, &result);
    tab_set(&ctx->output_pep_i_to_mic_pep_i, pep_i, &closest_mlpep_i);
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
    // dump_dyepeps(ctx);

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
        tab_ptr(DyeType, &ctx->dyemat, 0),
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
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        int ret = pthread_create(&ids[i], NULL, context_work_orders_worker, ctx);
        ensure(ret == 0, "Thread not created.");
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        pthread_join(ids[i], NULL);
    }
}
