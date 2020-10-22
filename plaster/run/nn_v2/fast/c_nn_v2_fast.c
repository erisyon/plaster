#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "math.h"
#include "c_nn_v2_fast.h"

// The following are alternative metrics
/*
float dist_inv_square(RadType *radrow, RadType *dyerow, Size n_cols) {
    RadType *rad = radrow;
    RadType *dye = dyerow;
    RadType sq_dist = (RadType)0;
    for (Index i=0; i<n_cols; i++) {
        RadType delta = *rad - *dye;
        sq_dist += delta * delta;
    }
    return (RadType)1 / sq_dist;
}


void score_weighted_inv_square(
    int n_neighbors,
    int *neighbor_dyt_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    float *neighbor_dists,  // array((n_neighbors,), type=float): distances computed by FLANN
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Tab *dyetrack_weights,  // arrays((n_dyetracks,), type=RadType): All dye weights
    Score *output_scores  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = neighbor_dyt_iz[nn_i];
        RadType neighbor_dist = neighbor_dists[nn_i];
        WeightType neighbor_weight = tab_get(WeightType, dyetrack_weights, neighbor_i);

        output_scores[nn_i] = (Score)(
            neighbor_weight / (0.1 + (neighbor_dist * neighbor_dist))
            // Add a small bias to avoid divide by zero
        );
    }
}
*/

void score_weighted_gaussian_mixture(
    NNV2FastContext *ctx,
    int n_neighbors,
    Size n_cols,
    int *neighbor_dyt_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    Tab *train_dyemat,  // arrays((n_dyetracks, n_cols), type=RadType): All dye weights
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Tab *dyetrack_weights,  // arrays((n_dyetracks,), type=RadType): All dye weights
    Score *output_scores  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    double weights[N_MAX_NEIGHBORS];
    double weighted_pdf[N_MAX_NEIGHBORS];
    double weighted_pdf_sum = 0.0;
    double std_per_dye = sqrt(0.1);

    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = neighbor_dyt_iz[nn_i];
        RadType *neighbor_target_dt = tab_ptr(RadType, train_dyemat, neighbor_i);
        WeightType neighbor_weight = tab_get(WeightType, dyetrack_weights, neighbor_i);
        weights[nn_i] = (double)neighbor_weight;

        double vdist = (double)0.0;
        double det = 1.0;
        for (Index col_i=0; col_i<n_cols; col_i++) {
            double target_dt_for_col_i = (double)neighbor_target_dt[col_i];
            double delta = (double)radrow[col_i] - target_dt_for_col_i;

            double std_units = std_per_dye * (target_dt_for_col_i == 0.0 ? 0.5 : target_dt_for_col_i);
            double variance = std_units * std_units;
            ensure_only_in_debug(variance > 0, "Illegal zero variance");
            det *= variance;
            vdist += delta * delta / variance;
        }
        ensure_only_in_debug(det > 0, "Illegal zero det");
        double inv_sqrt_det = 1.0 / sqrt(det);
        double pdf = inv_sqrt_det * exp(-vdist / 2.0);
        double wpdf = (double)neighbor_weight * pdf;
        weighted_pdf[nn_i] = wpdf;
        weighted_pdf_sum += wpdf;

        if(ctx->stop_requested) {
            return;
        }
    }

    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Score penalty = (Score)(1.0 - exp(-0.8 * weights[nn_i]));
        if(weighted_pdf_sum > 0.0) {
            Score score_pre_penalty = (Score)(weighted_pdf[nn_i] / weighted_pdf_sum);
            output_scores[nn_i] = penalty * score_pre_penalty;
        }
        else {
            output_scores[nn_i] = (Score)0;
        }

        if(ctx->stop_requested) {
            return;
        }
    }
}


Float64 normalCDF(Float64 value) {
   return 0.5 * erfc(-value * M_SQRT1_2);
}


// TODO: Research best way choose standard_n_dz
#define standard_n_dz 0.01
Float64 p_value_from_z_score(Float64 z_score) {
    return normalCDF(z_score + standard_n_dz) - normalCDF(z_score - standard_n_dz);
}


Float64 p_from_gaussian(Float64 x, Float64 mu, Float64 sigma) {
    Float64 z_score = (x - mu) / sigma;
    return p_value_from_z_score(z_score);
}


void score_k_fit_lognormal_mixture(
    NNV2FastContext *ctx,
    Size n_neighbors,
    Size n_cols,  // n_channels * n_cycles
    Tab *neighbor_dyt_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    Tab *train_dyemat,  // arrays((n_dyetracks, n_cols), type=RadType): All dye weights
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Tab *output_p_vals,  // array((n_neighbors,), type=float): returned scores for each neighbor
    Tab *output_pred_row_ks  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    // This is a log-normal model where the zero-counts (darks) are treated differently.
    // The non-zeros are log() and those mapping to zeros are not.
    // This requires that ctx contain beta, sigma for the lognormal and
    // zero_mu, zero_sigma for the normal of the zeros.

    Float64 beta = ctx->beta;
    Float64 sigma = ctx->sigma;
    Float64 zero_mu = ctx->zero_mu;
    Float64 zero_sigma = ctx->zero_sigma;

    for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = tab_get(Index, neighbor_dyt_iz, nn_i);
        RadType *target_dt = tab_ptr(RadType, train_dyemat, neighbor_i);

        RadType adjusted_radrow[N_MAX_CHANNELS * N_MAX_CYCLES];
        RowKType pred_k = 1.0;
        if(ctx->run_row_k_fit) {
            // If fitting the k value for the row then solve for pred_k
            // and write the adjusted radrow into adjusted_radrow.
            // And then swap out the radrow pointer for this adjustment.
            RowKType sum_of_radrow_squares = 0.0;
            RowKType sum_of_radrow_beta_dyerow_products = 0.0;
            for(Index col_i=0; col_i<n_cols; col_i++) {
                sum_of_radrow_squares += radrow[col_i] * radrow[col_i];
                sum_of_radrow_beta_dyerow_products += radrow[col_i] * (target_dt[col_i] * beta);
            }
            if(sum_of_radrow_beta_dyerow_products > 0) {
                pred_k = sum_of_radrow_squares / sum_of_radrow_beta_dyerow_products;
            }
            else {
                pred_k = 1.0;
            }
            for(Index col_i=0; col_i<n_cols; col_i++) {
                adjusted_radrow[col_i] = radrow[col_i] / pred_k;
            }
            radrow = adjusted_radrow;
        }

        Float64 p_value = 1.0; // This is an accumulated product
        for(Index col_i=0; col_i<n_cols; col_i++) {
            Float64 rad, z_score;
            if(target_dt[col_i] > 0) {
                rad = log(max(1e-50, (Float64)radrow[col_i]));
                z_score = (rad - beta) / sigma;
            }
            else {
                rad = (Float64)radrow[col_i];
                z_score = (rad - zero_mu) / zero_sigma;
            }
            p_value *= p_value_from_z_score(z_score);
        }

        tab_set(output_p_vals, nn_i, &p_value);
        tab_set(output_pred_row_ks, nn_i, &pred_k);

        if(ctx->stop_requested) {
            return;
        }
    }
}


void context_classify_radrows(
    NNV2FastContext *ctx,
    Tab radrows,
    Tab output_pred_pep_iz,
    Tab output_pred_dyt_iz,
    Tab output_scores,
    Tab output_dists   // Only used when ctx->run_against_all_dyetracks is true
) {
    Size n_cols = ctx->n_cols;
    Size n_rows = radrows.n_rows;
    ensure(n_rows <= 1024*16, "Too many rows (might overflow stack)");

    Size n_neighbors = 0;

    Tab *neighbor_dyt_iz = NULL;
    if(ctx->run_against_all_dyetracks) {
        // In this mode there is no neighbor look up
        // Each radrow is compared to every dyerow.
        // This is implemented by setting the neighbor_dyt_iz to point to every dyt
        Size n_dyts = ctx->train_dyemat.n_rows / n_cols;
        n_neighbors = n_dyts;
        Tab _neighbor_dyt_iz = tab_malloc_by_n_rows(n_neighbors, sizeof(int), TAB_NOT_GROWABLE);
        neighbor_dyt_iz = &_neighbor_dyt_iz;
        for(Index i=0; i<n_dyts; i++) {
            tab_set(neighbor_dyt_iz, i, &i);
        }
    }
    else {
        // In this mode we use FLANN to lookup candidate neighbors
        // TODO: This will need to sweep over pred_k
        n_neighbors = ctx->n_neighbors;
        Tab _neighbor_dyt_iz = tab_malloc_by_n_rows(n_rows * n_neighbors, sizeof(int), TAB_NOT_GROWABLE);
        neighbor_dyt_iz = &_neighbor_dyt_iz;

        // FETCH a batch of neighbors from FLANN in one call.
        if(ctx->n_threads > 1) {
            pthread_mutex_lock(&ctx->flann_index_lock);
        }
        ensure_only_in_debug(sizeof(int) == sizeof(Index), "int and Index not identical");
        flann_find_nearest_neighbors_index_float(
            ctx->flann_index_id,
            tab_ptr(RadType, &radrows, 0),
            n_rows,
            tab_ptr(int, neighbor_dyt_iz, 0),
            NULL, // Is this allowed?
            n_neighbors,
            &ctx->flann_params
        );
        if(ctx->n_threads > 1) {
            pthread_mutex_unlock(&ctx->flann_index_lock);
        }
    }

    Tab neighbor_p_vals = tab_malloc_by_n_rows(n_rows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);
    Tab neighbor_pred_row_ks = tab_malloc_by_n_rows(n_rows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);

    // Compare every radrow to the "neighbor" dytetracks.
    for (Index row_i=0; row_i<n_rows; row_i++) {
        if(ctx->stop_requested) {
            break;
        }

        RadType *radrow = tab_ptr(RadType, &radrows, row_i);

        Tab *row_neighbor_dyt_iz;
        if(ctx->run_against_all_dyetracks) {
            // In this mode the "neighbors" are actually ALL dyetracks.
            row_neighbor_dyt_iz = neighbor_dyt_iz;
        }
        else {
            // In this mode the "neighbors" come form FLANN
            Tab _row_neighbor_dyt_iz = tab_subset(neighbor_dyt_iz, row_i * n_neighbors, n_neighbors);
            row_neighbor_dyt_iz = &_row_neighbor_dyt_iz;
        }

        Tab row_neighbor_p_vals = tab_subset(&neighbor_p_vals, row_i * n_neighbors, n_neighbors);
        Tab neighbor_pred_row_ks = tab_subset(&neighbor_pred_row_ks, row_i * n_neighbors, n_neighbors);

        score_k_fit_lognormal_mixture(
            ctx,
            n_neighbors,
            n_cols,
            row_neighbor_dyt_iz,
            &ctx->train_dyemat,
            radrow,
            &row_neighbor_p_vals,
            &neighbor_pred_row_ks
        );

        // At this point there is a neighbor_p_vals for each neighbor dyt.
        // If ctx->run_row_k_fit is true then there is also a fit_k

        // Now each target score is computed by combining:
        //  * the p(target_dytrack) = how often this dyetrack is generated (the dyetrack weight)
        //  * the p(k) = the probability of the pred_k (assuming k fitting is enabled)
        //  * the rareness penalty (lowers score for assignment to very rare dyetracks)
        //  * The score is then normalized by the sum of all neighbors

        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Float64 p_val = tab_get(Float64, &row_neighbor_p_vals, nn_i);

            Index dyt_i = tab_get(Index, row_neighbor_dyt_iz, nn_i);
            WeightType target_weight = tab_get(WeightType, &ctx->train_dyetrack_weights, dyt_i);

            Score penalty = (Score)(1.0 - exp(-0.8 * target_weight));

            Float64 pred_row_k = tab_get(Float64, &neighbor_pred_row_ks, nn_i);
            Float64 p_row_k = p_from_gaussian(pred_row_k, 1.0, ctx->row_k_std);

            p_val *= penalty * target_weight * p_row_k;

            // UPDATE the p_vals with the score
            tab_set(&row_neighbor_p_vals, nn_i, &p_val);
        }

        // PICK target dyetrack with the highest score
        Float64 highest_score = (Float64)0;
        Float64 score_sum = (Float64)0;
        Index highest_score_i = 0;
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Float64 score = tab_get(Float64, &row_neighbor_p_vals, nn_i);
            if (score > highest_score) {
                highest_score = score;
                highest_score_i = nn_i;
            }
            score_sum += score;
        }

        Index most_likely_dyt_i = tab_get(Index, row_neighbor_dyt_iz, highest_score_i);
        Score dyt_score = highest_score / score_sum;

        // PICK peptide winner using Maximum Likelihood
        // the .pyx asserts that these are sorted by highest
        // count so we can just pick [0] from the correct dyepep
        Index dyepeps_offset = tab_get(Index, &ctx->train_dyt_i_to_dyepep_offset, most_likely_dyt_i);
        Index *dyepeps_block = tab_ptr(Index, &ctx->train_dyepeps, dyepeps_offset);
        ensure_only_in_debug(most_likely_dyt_i == 0 || dyepeps_block[0] == most_likely_dyt_i, "dyepeps_block points to wrong block");
        Index most_likely_pep_i = dyepeps_block[1];

        WeightType weight = tab_get(WeightType, &ctx->train_dyetrack_weights, most_likely_dyt_i);
        Score pep_score = (Score)dyepeps_block[2] / (Score)weight;
        Score score = dyt_score * pep_score;

        // Feels like this check could come out but it is safest as is.
        if(most_likely_dyt_i == 0) {
            score = (Score)0.0;
        }

        // Set output
        tab_set(&output_pred_dyt_iz, row_i, &most_likely_dyt_i);
        tab_set(&output_pred_pep_iz, row_i, &most_likely_pep_i);
        tab_set(&output_scores, row_i, &score);
    }

    tab_free(&neighbor_p_vals);
    tab_free(&neighbor_pred_row_ks);
    tab_free(neighbor_dyt_iz);
}


Index context_work_orders_pop(NNV2FastContext *ctx) {
    // NOTE: This return +1! So that 0 can be reserved.

    if(ctx->n_threads > 1) {
        pthread_mutex_lock(&ctx->work_order_lock);
    }

    Index i = ctx->next_row_i;
    ctx->next_row_i += ctx->n_rows_per_block;

    if(ctx->n_threads > 1) {
        pthread_mutex_unlock(&ctx->work_order_lock);
    }

    if(i < ctx->n_rows) {
        return i + 1;
    }
    return 0;
}

void progress_thread_safe(NNV2FastContext* ctx, int complete, int total, int retry) {
    if(ctx->n_threads > 1) {
        pthread_mutex_lock(&ctx->pyfunction_lock);
    }

    ctx->progress_fn(complete, total, retry);

    if(ctx->n_threads > 1) {
        pthread_mutex_unlock(&ctx->pyfunction_lock);
    }
}

typedef struct {
    pthread_t id;
    int complete;
    NNV2FastContext *ctx;
} ThreadContext;


void *context_work_orders_worker(void *_tctx) {
    // The worker thread. Pops off which pep to work on next
    // continues until there are no more work orders.
    ThreadContext* tctx= (ThreadContext *)_tctx;
    NNV2FastContext *ctx = tctx->ctx;
    while(!ctx->stop_requested) {
        Index row_i_plus_1 = context_work_orders_pop(ctx);
        if(row_i_plus_1 == 0) {
            break;
        }
        Index row_i = row_i_plus_1 - 1;
        context_classify_radrows(
            ctx,
            tab_subset(&ctx->test_radmat, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_pred_pep_iz, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_pred_dyt_iz, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_scores, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_dists, row_i, ctx->n_rows_per_block)
        );
        progress_thread_safe(ctx, row_i, ctx->n_rows, 0);
    }
    progress_thread_safe(ctx, ctx->n_rows, ctx->n_rows, 0);
    tctx->complete = 1;
    return (void *)0;
}


int context_start(NNV2FastContext *ctx) {
    ensure(sanity_check() == 0, "Sanity checks failed");
    ensure(
        ctx->n_neighbors <= ctx->train_dyemat.n_rows,
        "FLANN does not support requesting more neighbors than there are data points"
    );
    ensure(
        ctx->n_neighbors <= N_MAX_NEIGHBORS,
        "Too many neighbors requested"
    );

    // context_print(ctx);

    // CLEAR internally controlled elements
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;
    ctx->flann_index_id = 0;
    ctx->flann_params.cores = 1;
    ctx->stop_requested = 0;

    // CREATE the ANN index
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_float(
        tab_ptr(RadType, &ctx->train_dyemat, 0),
        ctx->train_dyemat.n_rows,
        ctx->n_cols,
        &speedup,
        &ctx->flann_params
    );

    ThreadContext tctxs[256];
    ensure(0 < ctx->n_threads && ctx->n_threads < 256, "Invalid n_threads");

    if(ctx->n_threads > 1) {
        int ret = pthread_mutex_init(&ctx->work_order_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");

        ret = pthread_mutex_init(&ctx->flann_index_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");

        ret = pthread_mutex_init(&ctx->pyfunction_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        tctxs[i].ctx = ctx;
        tctxs[i].complete = 0;
        int ret = pthread_create(&tctxs[i].id, NULL, context_work_orders_worker, &tctxs[i]);
        ensure(ret == 0, "Thread not created.");
    }

    int interrupted = 0;
    while(1) {
        int complete = 1;
        for(Index i=0; i<ctx->n_threads; i++) {
            if(!tctxs[i].complete) {
                complete = 0;
                break;
            }
        }

        if(ctx->n_threads > 1) {
            pthread_mutex_lock(&ctx->pyfunction_lock);
        }
        int got_interrupt = ctx->check_keyboard_interrupt_fn();
        if(ctx->n_threads > 1) {
            pthread_mutex_unlock(&ctx->pyfunction_lock);
        }

        if(got_interrupt) {
            printf("Ctrl-C received, please wait a few seconds until all threads complete\n");
            ctx->stop_requested = 1;
            interrupted = 1;
            break;
        }

        if(complete) {
            break;
        }
        // Calling ctx->check_keyboard_interrupt_fn too often results in a segfault
        usleep(50000);
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        pthread_join(tctxs[i].id, NULL);
    }

    return interrupted;
}


void context_free(NNV2FastContext *ctx) {
    if(ctx->flann_index_id) {
        flann_free_index_float(ctx->flann_index_id, &ctx->flann_params);
        ctx->flann_index_id = 0;
    }
}


void context_print(NNV2FastContext *ctx) {
    printf("n_neighbors=%ld\n", ctx->n_neighbors);
    printf("n_cols=%ld\n", ctx->n_cols);
    printf("run_against_all_dyetracks=%ld\n", ctx->run_against_all_dyetracks);
    printf("train_dyemat.n_rows=%ld\n", ctx->train_dyemat.n_rows);
    printf("test_radmat.n_rows=%ld\n", ctx->test_radmat.n_rows);
    for(Index row_i=0; row_i<ctx->test_radmat.n_rows; row_i++) {
        RadType *radrow = tab_ptr(RadType, &ctx->test_radmat, row_i);
        for(Index c=0; c<ctx->n_cols; c++) {
            printf("%2.1f ", radrow[c]);
        }
        printf("\n");
    }
}
